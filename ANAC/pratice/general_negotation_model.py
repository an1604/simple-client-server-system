import math
import random
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from negmas.sao import SAOResponse, SAONegotiator
from negmas import Outcome, ResponseType, SAOState


def initialize_DetReg(initial_value, reserved_value, time_high=1000, time_low=10) -> tuple:
    """
    The first initialization of the DetReg rectangle, for a later learning process of the opponent behaviour.
    Args:
         initial_value(float): The initial value of the DetReg rectangle, such as the lowest value that the agent can accept and
            the lower bound in the rectangle.
         reserved_value(float): The reserved value of the DetReg rectangle, like the upper bound in the rectangle, such as the
            best value that the agent can get.
         time_high(float): The upper bound in the context of time, because of this time dependent approach, we need to keep
            track of the time limit.
        time_low(float): The lower bound in the context of time, because of this time dependent approach, we need to keep
            track of the time limit.
    Returns:
        tuple: The initialized DetReg rectangle with (Tl, Th, Pl, Ph) values.
    """
    return (time_low, time_high, initial_value, reserved_value)


class GeneralNegotiationModel(SAONegotiator):
    def __init__(self, *args, **kwargs) -> None:
        """
                Initializes the general negotiation model.
                Make the following steps inside:
                1. Initialize the time variable to 0.0, to keep track of the time limit.
                2.  We initialize the DetReg tuple using an upper bound of the best utility and worst utility values as bounds.
        """
        super(GeneralNegotiationModel, self).__init__(*args, **kwargs)
        self.elimination_time = 1000  # TODO:SET TO THE CORRECT ELIMINATION TIME OF THE NEGOTIATION.
        self.opponent_times: list[float] = []
        self.opponent_utilities: list[float] = []
        self.previous_offers: list[tuple[float, float]] = []
        self._past_opponent_rv = 0.0
        self._future_opponent_rv = 0
        self.DetReg_tuple = initialize_DetReg(initial_value=self.opponent_ufun.worst(),
                                              reserved_value=self.opponent_ufun.best()
                                              )
        self.regression_calculations_dp: dict[str, float] = {}
        self.total_cells_as_clusters = None
        self.kmeans_clustering = None
        self.min_max_scaler = None
        self.probability_distribution_hypotheses: dict[
            str, float] = {}  # The probability distribution hypotheses for the learning process

    def __call__(self, state: SAOState):
        offer = state.current_offer
        if offer is not None:
            current_offer = self.save_offer(offer, state.relative_time)
            rand_offers = self.generate_random_offers()
            self.set_initial_probabilities()
            for rand_offer in rand_offers:
                # We're calculating the regression line till the relative time (current)
                # Based on the opponent's historical offers
                if rand_offer[1] < state.relative_time:
                    # Calculating the regression line (li) for the random offer.
                    # We use dynamic programming to save time and unnecessary calculations
                    reg_line_li = self.calc_regression_curve(rand_offer=rand_offer, current_offer=current_offer)
                    # Extract the fitted offers that match the regression line constraint.
                    fitted_offers = self.get_fitted_offers(regression_line=reg_line_li, random_offers=rand_offers)
                    if len(fitted_offers) > 0:
                        """
                        Calculating the non-linear correlation between opponent's historical offers and the fitted offers.
                        The more consistent the fitted offers are with opponent’s historical offers, 
                        the higher the conditional probability P(O|Hi) will be. 
                         """
                        coefficient = self.calc_nonlinear_correlation_coefficient(fitted_offers=fitted_offers,
                                                                                  t=state.relative_time)
                        # self.increase_conditional_probability_Hi()
                        self.update_hypothesises(current_offer=state.current_offer, conditional_porba=coefficient)
                        accept = self.is_acceptable(current_offer=state.current_offer,
                                                    reserved_offer=(self.elimination_time, self.ufun.best()),
                                                    rand_offer=rand_offer)
                        if accept:
                            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

    def save_offer(self, current_offer, relative_time):
        self.opponent_times.append(relative_time)
        self.opponent_utilities.append(self.opponent_ufun(current_offer))
        offer = (current_offer, relative_time, self.opponent_ufun.reserved_value)
        self.previous_offers.append(offer)
        self._past_opponent_rv = self.opponent_ufun.reserved_value
        return offer

    def generate_random_offers(self, time_grid_size=10, value_grid_size=5, ) -> list:
        """
        Generates random reservation points within each cell of the DetReg.
        At round tb, the buyer selects a random reservation point Xi (Ti(Xi),Pi(Xi)) in each cell Ci of the detecting region.

        Args:
            time_grid_size (int, optional): Size of the time grid cells. Default to 10.
            value_grid_size (int, optional): Size of the value grid cells. Default to 5.

        Returns:
            list: A list of randomly generated reservation points (time, price, reservation_value) tuples.
        ***
        Note: We make the cells inside the DetReg grid to be as clusters, using a clustering DBSCAN model that will find
            the relationships between the randon points and cluster them to 'cells'.
        ***
        """
        self.init_cluster_from_opponent_outcome_space()  # Initializing the cluster model and fit him
        # Define time and value grids
        time_start, time_end, offer_start, offer_end = self.DetReg_tuple
        time_cells = range(time_start, time_end, time_grid_size)
        price_cells = range(offer_start, offer_end, value_grid_size)
        num_of_generations = 20  # TODO: Adjusting this value to get better results
        random_offers = []
        for i in range(num_of_generations):
            for time_cell in time_cells:
                for price_cell in price_cells:
                    # Define cell boundaries (inclusive)
                    min_time = time_cell
                    max_time = min_time + time_grid_size - 1
                    min_price = price_cell
                    max_price = min_price + value_grid_size - 1

                    # Generate random point within the cell boundaries
                    rand_time = random.uniform(min_time, max_time)
                    rand_price = random.uniform(min_price, max_price)
                    rv = self.opponent_ufun(
                        (rand_time, rand_price))  # The reserved value from the opponent's utility function.
                    rand_point = (rand_time, rand_price, rv)
                    random_offers.append(rand_point)

        return random_offers

    def init_cluster_from_opponent_outcome_space(self, EPS_VAL=0.3, MIN_SAMPLES=3.5):
        """
        Creating the DetReg cells using the opponent's outcome space that already given.
        Args:
             EPS_VAL(float,optional) - The epsilon that mentioned the maximum distance between
                two samples for one to be considered as in the neighborhood of the other (for the clustering technique).
            MIN_SAMPLES (float,optional) - The number of samples (or total weight) in a neighborhood for
                a point to be considered as a core point.
        """
        self.total_cells_as_clusters = DBSCAN(eps=EPS_VAL, min_samples=MIN_SAMPLES)
        self.min_max_scaler = MinMaxScaler()  # feature scaling (between 0-1)

        rand_samples = set()
        for i in range(10):
            rand_samples.add(self.opponent_ufun.outcome_space.sample(n_outcomes=100, with_replacement=True))
        rand_samples = [
            outcome for outcome in rand_samples
            if self.in_DetReg_boundaries(outcome)
        ]
        X = np.array(rand_samples)

        self.min_max_scaler.fit(X)
        X_scaled = self.min_max_scaler.transform(X)
        self.total_cells_as_clusters.fit(X_scaled)  # Make clusters according to the outcome space sample
        # Initialize the KMEANS eith the number of clusters from the DBSCAN
        n_clusters = len(set(self.total_cells_as_clusters.labels_))  # Taking the number of clusters to the kmeans model
        self.kmeans_clustering = KMeans(n_clusters=n_clusters, init='k-means++')
        self.kmeans_clustering.fit(X_scaled)

    def predict_cluster(self, new_data_point):
        """
        Predicts the cluster label for a new data point using distance to core samples.

        Args:
            new_data_point: A new data point to be assigned to a cluster.

        Returns:
            int: The predicted cluster label for the new data point, or -1 if no close cluster is found.
        """

        # Get core samples and cluster labels (assuming they're stored)
        core_samples = self.total_cells_as_clusters.core_samples_
        cluster_labels = self.total_cells_as_clusters.labels_

        # Minimum distance and corresponding cluster (initialization)
        min_distance = np.inf
        predicted_cluster = -1

        # Loop through core samples and find the closest one
        for i, core_sample in enumerate(core_samples):
            distance = np.linalg.norm(new_data_point - core_sample)  # Calculate distance
            if distance < min_distance:
                min_distance = distance
                predicted_cluster = cluster_labels[i]

        return predicted_cluster

    def in_DetReg_boundaries(self, outcome):
        """
        Checks if a given outcome falls within the boundaries of the DetReg.
  
        Args:
            outcome: A single outcome value from the opponent's outcome space.
  
        Returns:
            bool: True if the outcome is within DetReg boundaries, False otherwise.
        """
        time_start, time_end, offer_start, offer_end = self.DetReg_tuple
        return (time_start <= outcome[0] <= time_end and
                offer_start <= outcome[1] <= offer_end)

    def set_initial_probabilities(self):
        # We use uniform distribution for now
        n_all = self.total_cells_as_clusters.n_clusters
        clusters_labels = self.total_cells_as_clusters.labels_
        for i, label in zip(n_all, set(clusters_labels)):
            k = f"H_{label}"
            self.probability_distribution_hypotheses[k] = 1 / n_all

    def probability_of_Hi(self, i):
        k = f'H_{i}'
        return self.probability_distribution_hypotheses[k]

    def get_probability_distribution_hypotheses(self, single_outcome, single_hypotheses):
        single_outcome = self.min_max_scaler(single_outcome)
        single_outcome = np.array(single_outcome)
        prediction = self.kmeans_clustering.predict([single_outcome])
        # TODO: CHECK THE OUTPUT OF `prediction`!
        if prediction == single_hypotheses:
            return self.probability_distribution_hypotheses[f"H_{prediction}"]

    def update_hypothesises(self, current_offer, conditional_porba):
        """
        The Bayesian learning process:
            returns a renewed belief based on the observed outcome O(current_offer) and at next round, the agent will
            update the prior probability P(Hi) using the posterior probability P(Hi|O), thus a
            more precise estimation is achieved by using the following Equation:
                P(Hi|O) = P(Hi)P(O|Hi) // ∑ (N_all> k >0) = P(O|Hk)P(Hk)
                Where:
                    P(O|Hi)- represents the likelihood that outcome might happen based on hypothesis Hi.
                    P(Hi) - represents the hypothesis probability.
                Args:
                    param current_offer: the current offer of the opponent.
        """
        denominator = 0.0
        for k, v in self.probability_distribution_hypotheses.items():
            # Calculate P(Hi)
            label = int(k.split('_')[-1])
            p_h_i = self.probability_of_Hi(label)

            # conditional_probability = self.get_probability_distribution_hypotheses(single_outcome=current_offer,
            #                                                                        single_hypotheses=label)
            denominator += p_h_i * conditional_porba

        n_all = [k for k in self.probability_distribution_hypotheses.keys()]
        for key_i in range(n_all):
            i = int(key_i.split('_')[-1])
            numerator = self.probability_of_Hi(i) * conditional_porba  # self.get_probability_distribution_hypotheses(
            # single_outcome = current_offer,
            # single_hypotheses = i)
            self.probability_distribution_hypotheses[key_i] = numerator / denominator

    def calc_regression_curve(self, rand_offer, current_offer) -> tuple:
        """
        The regression analysis process:
            At each random point that chosen in the __call__() method, the agent have to calculate
            the regression curve li based on the opponent's historical offers.
            Using this equation:
                offer(t) = p0 + (rv-px)(t/tix)^b
                which is:
                    p0 - The first offer in the historical offers.
                    pix - The current price of the random offer chosen.
                    t - The time of the random offer chosen.
                    Ti - The deadline negotiation time.
                    b - The concession parameter (will be found using another helper function).
            Args:
                rand_offer(tuple): The offer in which the regression line is calculated.
                current_offer(tuple): The current offer from the opponent.
            ***
            Note:
                We adopt Dynamic Programming in this case to save time,
                computational efficiency and performance.
            ***
        """
        rand_offer_key = self.generate_key_for_dp(offer=rand_offer, parent_offer=current_offer)
        if not rand_offer_key in self.regression_calculations_dp.keys():
            p0 = self.previous_offers[-1][1]
            tix = rand_offer[1]  # The offer's time in the second place in the tuple.
            pix = rand_offer[0]  # The offer itself is in the first place in the tuple.
            rv = rand_offer[2]  # The offer's reservation value.

            t = current_offer[0]  # The current offer's relative time
            pi = current_offer[1]  # The current offer's itself
            b = self.get_beta(p0=p0, tix=tix, t=t,
                              pi=pi, pix=pix)  # The concession parameter
            # The regression curve represents as tuple (time,bound (The curve itself))
            self.regression_calculations_dp[rand_offer_key] = (rand_offer[0], p0 + (rv - p0) * pow((t / tix), b))
        return self.regression_calculations_dp[rand_offer_key]

    def get_beta(self, p0, tix, t, pix, pi) -> float:
        """
        Returns the beta (The concession parameter) value from the following equation:
             b = sigma(
                    (ti*) * (pi*) for (ti*), (pi*) in self. previous_offers)
                    \\
                    (sigma((ti*) for (ti*) in self. previous_offers) ^ 2)
             where:
             pi* = ln((p0 - pi) \\ (p0 - pix))
             t* = ln(t // tix)
        """
        sum_of_tpi = 0.0
        sum_of_ti = 0.0
        for offer in self.previous_offers:
            if offer[0] < t:
                # pi*
                numerator = p0 - pi
                denominator = p0 - pix
                pi_star = math.log(numerator // denominator)  # Base e log

                # ti*
                t_star = math.log(t // tix)

                sum_of_tpi += pi_star * t_star
                sum_of_ti += pow(t_star, 2)
        return sum_of_tpi / sum_of_ti

    def get_fitted_offers(self, regression_line, random_offers) -> list:
        """
        Based on the calculated regression curve li,
        the agent can calculate the fitted offers Oˆ(tb) = {pˆ(0), pˆ(1),..., pˆ(tb) } at each round.
        """
        fitted_offers = []
        for previous_original_offer in self.previous_offers:
            for offer in random_offers:
                # We want to calculate the regression curve foreach offer
                li = self.calc_regression_curve(rand_offer=offer, current_offer=previous_original_offer)
                # If the regression line in fitted to the one in the params, it labels as fitted.
                if li[0] >= regression_line[0] and li[1] >= regression_line[1]:
                    fitted_offers.append(offer)
        return fitted_offers

    @staticmethod
    def generate_key_for_dp(offer: tuple, parent_offer: tuple) -> str:
        """Key for the Dynamic Programming purpose.
        We represent the key as a string, and the value as a float (represent the regression curve)
        The key is like:
            "p(parent offer information)__(target offer information)"
             Note that the reservation value of the two offers followed by 'rv' keyword.
        """
        return f"p:{parent_offer[0]}:{parent_offer[1]}:rv{parent_offer[2]}__{offer[0]}_{offer[1]}_rv{offer[2]}"

    def calc_nonlinear_correlation_coefficient(self, fitted_offers, t):
        """
        Calculate the nonlinear correlation between opponent’s historical offers O(tb) and the fitted offers Oˆ(tb).
        The coefficient of nonlinear correlation γ can be calculated by:
            γ = sigma((pi - p')(pi^ - (p^)'))
                \\
                sqrt(
                sigma((pi - p')^2
                *
                sigma((pi^) - (p^)'))^2
                )
        Where:
            (p^)' - The average value of all the fitted offers till time t.
            p' = The average value of all the historical offers of the opponent.
        ***
            Note: The non-linear correlation γ, where (0 ≤ γ ≤ 1), is a parameter reflecting the nonlinear similarity between the
            fitted offers and the resemblance between the random reservation point Xi and the opponent's real reservation point X.
        ***
        """
        # Calculate the average value of all fitted offers till time t (p^)'.
        average_fitted_offers = 0.0
        counter = 0
        for offer in fitted_offers:
            if offer[0] <= t:
                average_fitted_offers += offer[1]
                counter += 1
        average_fitted_offers = average_fitted_offers // counter

        # Calculating the average value of all the historical offers of the opponent (p').
        average_historical_offers = 0.0
        for offer in self.previous_offers:
            average_historical_offers += offer[1]
        average_historical_offers = average_historical_offers // len(self.previous_offers)

        # Calculating the Numerator -> ∑(tb>i>0) = (pi − p')(pˆi − (pˆ)')
        numerator = 0.0
        for i in range(t):
            pi = self.previous_offers[i][1]  # The value of the offer in time i(pi)
            rv_i = self.previous_offers[i][2]  # Reservation value for historical offer i
            pi_fitted = fitted_offers[i][1]  # The value of the fitted offer in tine i (p^i)
            numerator += (pi - average_historical_offers - (rv_i - average_historical_offers)) * \
                         (pi_fitted - average_fitted_offers)

        # Calculating the Denominator -> ∑ (tb>i>0) = (pi − p')^2 *  ∑ (n>i>0) = (pˆi − (pˆ)')^2
        n = len(fitted_offers)
        denominator = 0.0
        for i in range(t):
            a = 0.0
            pi = self.previous_offers[i][1]  # The value of the offer in time i(pi)
            a += pi - average_historical_offers
            a = pow(a, 2)
            b = 0.0
            for j in range(n):
                pi_fitted = fitted_offers[i][1]  # The value of the fitted offer in tine i (p^i)
                b += pi_fitted - average_historical_offers
                b = pow(b, 2)
            denominator += a * b
        denominator = math.sqrt(denominator)

        return numerator / denominator  # The correlation value at the end.

    def is_acceptable(self, current_offer, reserved_offer, rand_offer):
        """
            The concession strategy, we consider three different scenarios,
            and in each scenario,
            the buyer needs to adopt different concession strategies to maximize its expected utility.
            Args:
                current_offer (tuple): The agent's current offer.
                reserved_offer (tuple): The agent's reservation offer at the deadline.
                rand_offer (tuple): The random reservation offer of opponent.
        """
        p0 = self.previous_offers[-1][1]
        tix = rand_offer[0]
        Tb = reserved_offer[0]
        pix = rand_offer[1]
        t = current_offer[0]
        pi = current_offer[1]

        # First scenario - the agent's random reservation point is in the negotiation region
        if tix < Tb and pix > p0:
            return rand_offer
        # Second scenario - the random reservation point is out of the agent’s negotiation region
        elif tix >= Tb and pix >= p0:
            reg_line = self.calc_regression_curve(rand_offer=rand_offer, current_offer=current_offer)
            # We need to check if the regression curve pass through the negotiation region in any case
            if self.is_crosses_region(regression_curve=reg_line):
                # If the regression curve is in the negotiation region,
                # we need to check if the random offer is on the right side of the regression curve
                if tix >= reg_line[0] and pix >= reg_line[1]:
                    p1_concession_point = (
                        reserved_offer[0] - 1, reserved_offer[1])  # (T(b-1),rv) as the concession point
                    return p1_concession_point
                b0 = self.get_beta(p0=p0, tix=tix, t=t, pi=pi, pix=pix)
                phi = np.random.uniform(0.99, 1.0)  # φ(max) has to be quite close to 1.
                p2_concession_point = (b0 + 1, phi * self.DetReg_tuple[3])  # (b0 + 1,φ(max) · RPb)
                return p2_concession_point
        # Third scenario
        elif rand_offer[0] < reserved_offer[0] and rand_offer[1] < current_offer[1]:
            reg_line = self.calc_regression_curve(rand_offer=rand_offer, current_offer=current_offer)
            if self.is_crosses_bottom_line_region(regression_curve=reg_line):
                # set the concession point P1 be the point one step earlier than the
                # intersection point on the regression line.
                p1_concession_point = self.previous_offers[-1]
                return p1_concession_point
            # price will keep almost the same as the current price p0
            phi = random.random() * 0.001  # φ(min) need to be quite close to 0
            p2_concession_point = (rand_offer[0], (1 + phi) * p0)  # (tix,(1 + φ(min)· p0)
            return p2_concession_point

        elif tix >= Tb and pix <= p0:
            reg_line = self.calc_regression_curve(rand_offer=rand_offer, current_offer=current_offer)
            if self.is_crosses_region(regression_curve=reg_line):
                if tix >= reg_line[0] and pix >= reg_line[1]:
                    p1_concession_point = (
                        reserved_offer[0] - 1, reserved_offer[1])  # (T(b-1),rv) as the concession point
                    return p1_concession_point
            elif self.is_crosses_bottom_line_region(regression_curve=reg_line):
                p1_concession_point = self.previous_offers[-1]
                return p1_concession_point
            elif rand_offer[0] < reserved_offer[0] and rand_offer[1] < current_offer[1]:
                phi = random.random() * 0.001  # φ(min) need to be quite close to 0
                p2_concession_point = (rand_offer[0], (1 + phi) * p0)  # (tix,(1 + φ(min)· p0)
                return p2_concession_point
            else:
                b0 = self.get_beta(p0=p0, tix=tix, t=t, pi=pi, pix=pix)
                phi = np.random.uniform(0.99, 1.0)  # φ(max) has to be quite close to 1.
                p2_concession_point = (b0 + 1, phi * self.DetReg_tuple[3])  # (b0 + 1,φ(max) · RPb)
                return p2_concession_point

    def is_crosses_region(self, regression_curve) -> bool:
        # TODO: CHECK IF THE CONDITIONS ARE CORRECT.
        return self.DetReg_tuple[0] <= regression_curve[0] <= self.DetReg_tuple[1] \
            and self.DetReg_tuple[2] <= regression_curve[1] <= self.DetReg_tuple[3]

    def is_crosses_bottom_line_region(self, regression_curve):
        return self.DetReg_tuple[0] <= regression_curve[0]

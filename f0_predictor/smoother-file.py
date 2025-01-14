# f0_predictor/smoother.py
import numpy as np
from scipy.stats import norm

class ViterbiF0Smoother:
    def __init__(self, max_jump: float = 35.0, voicing_threshold: float = 0.5):
        self.max_jump = max_jump
        self.voicing_threshold = voicing_threshold

    def _calculate_transition_probability(self, f0_prev: float, f0_curr: float) -> float:
        if f0_prev == 0 or f0_curr == 0:
            return 0.5 if f0_prev == f0_curr else 0.1

        jump = abs(f0_curr - f0_prev)
        prob = norm.pdf(jump, 0, self.max_jump/5)
        return prob

    def _calculate_emission_probability(self, predicted_f0: float, observed_f0: float) -> float:
        if observed_f0 == 0:
            return 0.1 if predicted_f0 == 0 else 0.01

        std_dev = 10.0
        prob = norm.pdf(observed_f0, predicted_f0, std_dev)
        return prob

    def smooth(self, predictions: np.ndarray, original_f0: np.ndarray) -> np.ndarray:
        N = len(predictions)
        f0_min, f0_max = 50, 600
        step = 1
        states = np.append(np.arange(f0_min, f0_max, step), 0)
        
        V = np.full((len(states), N), -np.inf)
        B = np.zeros((len(states), N), dtype=np.int32)
        
        emission_probs = np.array([self._calculate_emission_probability(state, original_f0[0])
                                 for state in states])
        V[:, 0] = np.log(emission_probs + 1e-10)

        for t in range(1, N):
            curr_f0 = original_f0[t]
            transition_probs = np.array([self._calculate_transition_probability(prev_state, curr_f0)
                                       for prev_state in states])
            emission_probs = np.array([self._calculate_emission_probability(state, curr_f0)
                                     for state in states])
            
            log_trans = np.log(transition_probs + 1e-10)
            log_emit = np.log(emission_probs + 1e-10)

            for i in range(len(states)):
                probs = V[:, t-1] + log_trans + log_emit[i]
                max_idx = np.argmax(probs)
                V[i, t] = probs[max_idx]
                B[i, t] = max_idx

        smoothed_f0 = np.zeros(N)
        current_state = np.argmax(V[:, -1])
        smoothed_f0[-1] = states[current_state]

        for t in range(N-2, -1, -1):
            current_state = B[current_state, t+1]
            smoothed_f0[t] = states[current_state]

        return smoothed_f0
"""
RSA Baseline Model
Implements the Rational Speech Act framework for:
  1. Scalar Implicature (Frank & Goodman, 2012)
  2. Polite Speech (Yoon et al., 2020 / Murthy et al., 2025)
"""

import numpy as np
from itertools import product


def normalize(arr: np.ndarray) -> np.ndarray:
    """Row-wise normalization (softmax-free version)."""
    arr = np.array(arr, dtype=float)
    if arr.ndim == 1:
        total = arr.sum()
        return arr / total if total > 0 else np.ones_like(arr) / len(arr)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return arr / row_sums


class ScalarImplicatureRSA:
    """
    Referential communication game RSA model.

    Params
    ------
    objects : list of str
        All objects in the scene (e.g. ['red_square', 'red_circle', 'blue_square'])
    utterances : list of str
        Available utterances (e.g. ['red', 'square', 'blue', 'circle'])
    lexicon : np.ndarray, shape (n_utterances, n_objects)
        Boolean/float matrix: lexicon[u, o] = 1 if utterance u is true of object o
    prior : np.ndarray, shape (n_objects,)
        Prior probability that each object is the target
    alpha : float
        Rationality parameter (speaker optimality). Default=1.
    cost : np.ndarray or None
        Per-utterance cost vector. If None, assumed uniform (zero cost).
    """

    def __init__(self, objects, utterances, lexicon, prior=None, alpha=1.0, cost=None):
        self.objects = objects
        self.utterances = utterances
        self.lexicon = np.array(lexicon, dtype=float)
        n_obj = len(objects)
        self.prior = normalize(prior) if prior is not None else np.ones(n_obj) / n_obj
        self.alpha = alpha
        self.cost = np.array(cost, dtype=float) if cost is not None else np.zeros(len(utterances))

    def literal_listener(self) -> np.ndarray:
        """
        L0(obj | utt) ∝ [[utt is true of obj]] * prior(obj)
        Returns shape (n_utt, n_obj)
        """
        L0 = self.lexicon * self.prior[np.newaxis, :]
        return normalize(L0)

    def pragmatic_speaker(self) -> np.ndarray:
        """
        S1(utt | obj) ∝ exp(alpha * log L0(obj | utt) - cost(utt))
        Returns shape (n_obj, n_utt)
        """
        L0 = self.literal_listener()
        log_L0 = np.log(np.clip(L0, 1e-10, 1))
        utility = self.alpha * log_L0.T - self.cost[np.newaxis, :]
        utility -= utility.max(axis=1, keepdims=True)
        S1 = np.exp(utility)
        return normalize(S1)

    def pragmatic_listener(self) -> np.ndarray:
        """
        L1(obj | utt) ∝ S1(utt | obj) * prior(obj)
        Returns shape (n_utt, n_obj)
        """
        S1 = self.pragmatic_speaker()
        L1 = S1.T * self.prior[np.newaxis, :]
        return normalize(L1)

    def summary(self):
        """Print readable tables for L0, S1, L1."""
        import pandas as pd
        L0 = self.literal_listener()
        S1 = self.pragmatic_speaker()
        L1 = self.pragmatic_listener()
        print("=== L0: Literal Listener P(obj | utt) ===")
        print(pd.DataFrame(L0, index=self.utterances, columns=self.objects).round(3))
        print("\n=== S1: Pragmatic Speaker P(utt | obj) ===")
        print(pd.DataFrame(S1, index=self.objects, columns=self.utterances).round(3))
        print("\n=== L1: Pragmatic Listener P(obj | utt) ===")
        print(pd.DataFrame(L1, index=self.utterances, columns=self.objects).round(3))


class PoliteSpeechRSA:
    """
    Polite speech RSA model.

    The speaker has two goals:
      - Informational goal: communicate true state
      - Social goal: make listener feel good (positive face)

    Params
    ------
    states : list
        True states (e.g. [1, 2, 3, 4, 5] stars)
    utterances : list
        Possible utterances (e.g. ['terrible','bad','okay','good','amazing', ...])
    semantics : np.ndarray, shape (n_utt, n_states)
        P(state | utterance) under literal semantics
    alpha : float
        Rationality / optimality
    omega_i : float
        Weight on informational goal (0–1)
    omega_s : float
        Weight on social goal (0–1); typically omega_i + omega_s = 1
    prior : np.ndarray or None
        Prior over states
    """

    YOON_UTTERANCES = [
        "It's terrible",
        "It's bad",
        "It's okay",
        "It's good",
        "It's amazing",
        "It's not terrible",
        "It's not bad",
        "It's not okay",
        "It's not good",
        "It's not amazing",
    ]

    MURTHY_UTTERANCES = [
        "It's terrible",
        "It's bad",
        "It's okay",
        "It's good",
        "It's amazing",
        "It's not terrible",
        "It's not bad",
        "It's not good",
    ]

    def __init__(self, states, utterances, semantics, alpha=1.0,
                 omega_i=0.5, omega_s=0.5, prior=None):
        self.states = states
        self.utterances = utterances
        self.semantics = np.array(semantics, dtype=float)
        self.alpha = alpha
        self.omega_i = omega_i
        self.omega_s = omega_s
        n_states = len(states)
        self.prior = normalize(prior) if prior is not None else np.ones(n_states) / n_states

    def literal_listener(self) -> np.ndarray:
        """
        L0(state | utt) ∝ semantics(utt, state) * prior(state)
        Returns (n_utt, n_states)
        """
        L0 = self.semantics * self.prior[np.newaxis, :]
        return normalize(L0)

    def social_value(self) -> np.ndarray:
        """
        V_social(utt, true_state) = E_{L0(state|utt)}[state]  (expected welfare)
        Returns (n_utt,)  — depends only on utterance, not true state
        """
        L0 = self.literal_listener()
        state_vals = np.array(self.states, dtype=float)
        return L0 @ state_vals

    def informational_value(self) -> np.ndarray:
        """
        V_info(utt, true_state) = log L0(true_state | utt)
        Returns (n_states, n_utt)
        """
        L0 = self.literal_listener()
        return np.log(np.clip(L0, 1e-10, 1)).T

    def pragmatic_speaker(self) -> np.ndarray:
        """
        S1(utt | true_state) ∝ exp(alpha * [omega_i * V_info + omega_s * V_social])
        Returns (n_states, n_utt)
        """
        V_info = self.informational_value()
        V_soc  = self.social_value()
        utility = self.alpha * (
            self.omega_i * V_info +
            self.omega_s * V_soc[np.newaxis, :]
        )
        utility -= utility.max(axis=1, keepdims=True)
        S1 = np.exp(utility)
        return normalize(S1)

    def summary(self):
        import pandas as pd
        S1 = self.pragmatic_speaker()
        print(f"=== Polite S1 (omega_i={self.omega_i}, omega_s={self.omega_s}) ===")
        print(pd.DataFrame(S1, index=self.states, columns=self.utterances).round(3))


def get_frank_goodman_stimuli():
    """
    Referential game scenes following Frank & Goodman (2012).

    Key design: targets vary across scenes so the model cannot default
    to one word. The pragmatically correct answer differs by scene.

    Scene 1: Target = red_square  -> "red" is unique, "square" is ambiguous
    Scene 2: Target = blue_circle -> "circle" is unique, "blue" is ambiguous
    Scene 3: Target = red_square  -> two objects, "red" uniquely identifies
    Scene 4: Target = blue_square, skewed prior -> tests prior effect
    """
    scenes = []

    objects_3way    = ['blue_square', 'red_square', 'blue_circle']
    utterances_3way = ['blue', 'square', 'red', 'circle']
    lexicon_3way    = np.array([
        [1,    1,   0,   0],
        [0,    1,   1,   0],
        [1,    0,   0,   1],
    ]).T

    scenes.append(dict(
        name="Scene 1 (target=red_square, unique=red)",
        objects=objects_3way,
        utterances=utterances_3way,
        lexicon=lexicon_3way,
        prior=np.array([1/3, 1/3, 1/3]),
        target_idx=1,
        expected_utterance="red",
    ))

    scenes.append(dict(
        name="Scene 2 (target=blue_circle, unique=circle)",
        objects=objects_3way,
        utterances=utterances_3way,
        lexicon=lexicon_3way,
        prior=np.array([1/3, 1/3, 1/3]),
        target_idx=2,
        expected_utterance="circle",
    ))

    objects_2way    = ['red_square', 'blue_square']
    utterances_2way = ['red', 'blue', 'square']
    lexicon_2way    = np.array([
        [1, 0, 1],
        [0, 1, 1],
    ]).T

    scenes.append(dict(
        name="Scene 3 (two-object, target=red_square, unique=red)",
        objects=objects_2way,
        utterances=utterances_2way,
        lexicon=lexicon_2way,
        prior=np.array([0.5, 0.5]),
        target_idx=0,
        expected_utterance="red",
    ))

    scenes.append(dict(
        name="Scene 4 (target=blue_square, skewed prior)",
        objects=objects_3way,
        utterances=utterances_3way,
        lexicon=lexicon_3way,
        prior=np.array([0.6, 0.2, 0.2]),
        target_idx=0,
        expected_utterance="blue",
    ))

    return scenes

def get_yoon_semantics(states=None, utterances=None):
    """
    Approximate semantics matrix for polite speech stimuli.
    Based on Yoon et al. (2020) adjective scales.

    Returns states, utterances, semantics (n_utt, n_states).
    """
    if states is None:
        states = [1, 2, 3, 4, 5]
    if utterances is None:
        utterances = PoliteSpeechRSA.MURTHY_UTTERANCES

    def threshold_semantics(utt, state):
        thresholds = {
            "It's terrible": state <= 1,
            "It's bad":      state <= 2,
            "It's okay":     state >= 3,
            "It's good":     state >= 4,
            "It's amazing":  state >= 5,
            "It's not terrible": state > 1,
            "It's not bad":      state > 2,
            "It's not good":     state < 4,
        }
        return float(thresholds.get(utt, 0.5))

    n_utt    = len(utterances)
    n_states = len(states)
    sem      = np.zeros((n_utt, n_states))
    for i, utt in enumerate(utterances):
        for j, s in enumerate(states):
            sem[i, j] = threshold_semantics(utt, s)

    sem = sem + 0.01
    return states, utterances, sem


if __name__ == "__main__":
    print("=" * 60)
    print("SCALAR IMPLICATURE RSA — Frank & Goodman (2012)")
    print("=" * 60)
    scenes = get_frank_goodman_stimuli()
    for scene in scenes:
        print(f"\n--- {scene['name']} ---")
        rsa = ScalarImplicatureRSA(
            objects=scene['objects'],
            utterances=scene['utterances'],
            lexicon=scene['lexicon'],
            prior=scene['prior'],
        )
        rsa.summary()

    print("\n" + "=" * 60)
    print("POLITE SPEECH RSA — Yoon et al. (2020)")
    print("=" * 60)
    states, utterances, semantics = get_yoon_semantics()
    for omega_i, omega_s in [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]:
        rsa = PoliteSpeechRSA(
            states=states,
            utterances=utterances,
            semantics=semantics,
            alpha=3.0,
            omega_i=omega_i,
            omega_s=omega_s,
        )
        rsa.summary()

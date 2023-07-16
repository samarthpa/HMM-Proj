import unittest
import platform
import hashlib

if __name__ == "__main__":
    from HMM import part_1_a, part_2_a
    from HMM import viterbi, multidimensional_viterbi


if platform.system() == 'Windows':
    NIX = False
    print("Test on Windows system")
else:
    NIX = True
    print("Test on Linux/OS X system")

def print_success_message(test_case):
    print("UnitTest {0} passed successfully!".format(test_case))

class TestPart1a(unittest.TestCase):        

    def test_prior(self, part_1_a):
        (a_prior_probs, a_transition_probs, a_emission_paras,
        n_prior_probs, n_transition_probs, n_emission_paras,
        s_prior_probs, s_transition_probs, s_emission_paras) = part_1_a()

        a_prior = sum(a_prior_probs.values())
        n_prior = sum(n_prior_probs.values())
        s_prior = sum(s_prior_probs.values())
        total_prob = a_prior + n_prior + s_prior 
        msg = ('incorrect prior probs. each word should be selected with '
               'equal probability. counted {}, should be 1').format(total_prob)
        self.assertAlmostEqual(1.0, total_prob, places=2, msg=msg)
        print_success_message("test_prior")

    def test_a_emission(self, part_1_a):
        (a_prior_probs, a_transition_probs, a_emission_paras,
        n_prior_probs, n_transition_probs, n_emission_paras,
        s_prior_probs, s_transition_probs, s_emission_paras) = part_1_a()
        
        mean, std = a_emission_paras['A1']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for word ALLIGATOR, state A1'
        self.assertEqual("dc4e642e57357e0f1cb1b5d00322eaa03504e5345086df408e00c535d92c5c1c", mean_hash, msg)
        msg = 'incorrect std for word ALLIGATOR, state A1'
        self.assertEqual("7b40b9d0cc4a93e70b404b31441d8b6ad4baa9be85cdf1c87ac68b2dc2cac4e2", std_hash, msg)

        mean, std = a_emission_paras['A2']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for word ALLIGATOR, state A2'
        self.assertEqual("d848b818c065e37696a5a83b31f90b150dada7aff3159e9a801f9eab3f84543d", mean_hash, msg)
        msg = 'incorrect std for word ALLIGATOR, state A2'
        self.assertEqual("71b38a71dadf474805c1fb838ab0aac30a2be73fb8ae540990f362099ec0458a", std_hash, msg)

        mean, std = a_emission_paras['A3']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for word ALLIGATOR, state A3'
        self.assertEqual("c6fdb4c7c781483ed5058275d45bc68a335126c34bfb1849b63d0875b2666e82", mean_hash, msg)
        msg = 'incorrect std for word ALLIGATOR, state A3'
        self.assertEqual("de918c34fd162b7235733b7c81d76b202b010c490944a2e360ba275904a3b0cd", std_hash, msg)
        
        print_success_message("test_a_emission")

    def test_n_emission(self, part_1_a):
        (a_prior_probs, a_transition_probs, a_emission_paras,
        n_prior_probs, n_transition_probs, n_emission_paras,
        s_prior_probs, s_transition_probs, s_emission_paras) = part_1_a()

        mean, std = n_emission_paras['N1']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for word NUTS, state N1'
        self.assertEqual("af3861357513c7595874a57074a58a98526e1cb8ad5a71e8497bcb5b857f4ae2", mean_hash, msg)
        msg = 'incorrect std for word NUTS, state N1'
        self.assertEqual("aea25c614f99bcddaa58a6178a99569d16b5340ed0a6c96901c6723cb1a66e45", std_hash, msg)

        mean, std = n_emission_paras['N2']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for word NUTS, state N2'
        self.assertIn(mean_hash, ["53519e43db90bd08ff4459fd23fc944324ffb7d8f542ccc0b44257afea2ef525", "73475cb40a568e8da8a045ced110137e159f890ac4da883b6b17dc651b3a8049"], msg)
        msg = 'incorrect std for word NUTS, state N2'
        self.assertEqual("4dacbdf481f77b7385d1c5f286f306a3ef539dff02a0d0dbfccc17787705d0d0", std_hash, msg)

        mean, std = n_emission_paras['N3']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for word NUTS, state N3'
        self.assertIn(mean_hash, ["db58b6c40698d7371bbcff35d085e1bac5fa439d0de31eb8e0da7c47d27cb2a7", "39fa9ec190eee7b6f4dff1100d6343e10918d044c75eac8f9e9a2596173f80c9"], msg)
        msg = 'incorrect std for word NUTS, state N3'
        self.assertEqual("a0ac9b3dd38eecb310fa7e583d5ee2bce6236929a5e448d3d3c6b7c13a11900c", std_hash, msg)

        print_success_message("test_n_emission")

    def test_s_emission(self, part_1_a):
        (a_prior_probs, a_transition_probs, a_emission_paras,
        n_prior_probs, n_transition_probs, n_emission_paras,
        s_prior_probs, s_transition_probs, s_emission_paras) = part_1_a()
        
        mean, std = s_emission_paras['S1']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for word SLEEP, state S1'
        self.assertEqual("4defe1195908e76e51b32b81be42e655866e945ee168ece57bc5d53f1d0cf19e", mean_hash, msg)
        msg = 'incorrect std for word SLEEP, state S1'
        self.assertEqual("f975422fc861c785d6c6344981c501b2d3bda4d03e0d9f324481ef48c5f3e19f", std_hash, msg)

        mean, std = s_emission_paras['S2']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for word SLEEP, state S2'
        self.assertEqual("36b74139bd0ea465f5a1062708326be6402674312779940be0b99ba62d9f8cf3", mean_hash, msg)
        msg = 'incorrect std for word SLEEP, state S2'
        self.assertEqual("0d921877e5dd1b8b0458b49f422864b18180b6fb3ddc6b09c084f63f6fa661f9", std_hash, msg)

        mean, std = s_emission_paras['S3']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for word SLEEP, state S3'
        self.assertEqual("b7fc4a01285fb271d40e71eb0edde2b81de0df4282f24b657723559e9a6e0746", mean_hash, msg)
        msg = 'incorrect std for word SLEEP, state S3'
        self.assertEqual("d4e5ecf40ba5700a6c7c4a8ecac409c04f0bb0c85645e22e8a1899615637a649", std_hash, msg)

        print_success_message("test_s_emission")

    def test_a_transition(self, part_1_a):
        (a_prior_probs, a_transition_probs, a_emission_paras,
        n_prior_probs, n_transition_probs, n_emission_paras,
        s_prior_probs, s_transition_probs, s_emission_paras) = part_1_a()

        for state, probs in a_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('ALLIGATOR transition prob in state {} '
                   'should sum to 1 (got {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)
        print_success_message("test_a_transition")

    def test_n_transition(self, part_1_a):
        (a_prior_probs, a_transition_probs, a_emission_paras,
        n_prior_probs, n_transition_probs, n_emission_paras,
        s_prior_probs, s_transition_probs, s_emission_paras) = part_1_a()

        for state, probs in n_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('NUTS transition prob in state {} should sum to 1 '
                   '(got {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)
        print_success_message("test_n_transition")

    def test_s_transition(self, part_1_a):
        (a_prior_probs, a_transition_probs, a_emission_paras,
        n_prior_probs, n_transition_probs, n_emission_paras,
        s_prior_probs, s_transition_probs, s_emission_paras) = part_1_a()

        for state, probs in s_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('SLEEP transition prob in state {} should sum to 1 '
                   '(got {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)
        print_success_message("test_s_transition")

class TestPart1b(unittest.TestCase):

    def setup(self, part_1_a):
        a_states = ['A1', 'A2', 'A3', 'Aend']
        n_states = ['N1', 'N2', 'N3', 'Nend']
        s_states = ['S1', 'S2', 'S3', 'Send']

        (a_prior_probs, a_transition_probs, a_emission_paras,
        n_prior_probs, n_transition_probs, n_emission_paras,
        s_prior_probs, s_transition_probs, s_emission_paras) = part_1_a()

        states = a_states + n_states + s_states
        prior = a_prior_probs
        prior.update(n_prior_probs)
        prior.update(s_prior_probs)

        trans = a_transition_probs
        trans.update(n_transition_probs)
        trans.update(s_transition_probs)

        emiss = a_emission_paras
        emiss.update(n_emission_paras)
        emiss.update(s_emission_paras)
        return states, prior, trans, emiss

    def test_viterbi_case1(self, part_1_a, viterbi):
        evidence = []
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        msg = ('when evidence is an empty list, return "None" or [], '
                'got {}').format(seq)
        self.assertTrue(seq in [None, []], msg)
        msg = ('when evidence is an empty list, return prob=0.0, '
                'got {}').format(prob)
        self.assertTrue(prob == 0., msg)
        print_success_message("test_viterbi_case1")

    def test_viterbi_case2(self, part_1_a, viterbi):
        evidence = [30]
        prob_ans = 0.01576664562875057
        seq_ans = ['S1']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=7)
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_case2")


    def test_viterbi_case3(self, part_1_a, viterbi):
        evidence = [40]
        prob_ans = 0.011713950611283535
        seq_ans = ['N1']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=7)
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_case3")


    def test_viterbi_realsample1(self, part_1_a, viterbi):
        """
        Extracted from GISLR dataset: idx: 17; length: 8
        Actual words: ALLIGATOR
        """
        evidence = [20, 65, 20, 30, 45, 60, 60, 42]
        prob_ans = 5.739096406102214e-17
        seq_ans = ['A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=21)
        # print(seq)͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏󠄇͏︀
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_realsample1")


    def test_viterbi_realsample2(self, part_1_a, viterbi):
        """
        Extracted from GISLR dataset: idx: 8; length: 9
        Actual words: NUTS
        """
        evidence = [45, 35, 34, 44, 41, 42, 45, 46, 45]
        prob_ans = 4.968812725589942e-15
        seq_ans = ['N1', 'N1', 'N1', 'N1', 'N1', 'N1', 'N1', 'N1', 'N1']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=19)
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_realsample2")

    def test_viterbi_realsample3(self, part_1_a, viterbi):
        """
        Extracted from GISLR dataset: idx: 9; length: 12
        Actual words: SLEEP
        """
        evidence = [26, 22, 13, 26, 20, 31, 32, 39, 41, 42, 38, 40]
        prob_ans = 8.189507039078366e-20
        seq_ans = ['S1', 'S1', 'S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2', 'S2']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=24)
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_realsample3")


if __name__ == "__main__":
    TestPart1a().test_prior(part_1_a)
    TestPart1a().test_a_emission(part_1_a)
    TestPart1a().test_n_emission(part_1_a)
    TestPart1a().test_s_emission(part_1_a)
    TestPart1a().test_a_transition(part_1_a)
    TestPart1a().test_n_transition(part_1_a)
    TestPart1a().test_s_transition(part_1_a)
    TestPart1b().test_viterbi_case1(part_1_a, viterbi)
    TestPart1b().test_viterbi_case2(part_1_a, viterbi)
    TestPart1b().test_viterbi_case3(part_1_a, viterbi)
    TestPart1b().test_viterbi_realsample1(part_1_a, viterbi)
    TestPart1b().test_viterbi_realsample2(part_1_a, viterbi)
    TestPart1b().test_viterbi_realsample3(part_1_a, viterbi)



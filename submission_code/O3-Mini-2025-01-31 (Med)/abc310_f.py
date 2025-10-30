def main():
    import sys
    mod = 998244353
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # We need to count the number of dice‐roll outcomes (configuration)
    # for which there is some subset of dice (using only those dice that rolled a value ≤ 10)
    # whose sum is exactly 10.
    #
    # For each die i the outcome is uniformly chosen from 1 to A[i]. Notice that if the outcome is greater than 10,
    # then that die cannot contribute to a sum of 10 (you cannot add a number > 10 and still get exactly 10).
    #
    # We can “simulate” building up the set of achievable subset‐sums (which are numbers between 0 and 10)
    # as we go through the dice. We represent the set as a bitmask of 11 bits (bit j set when sum j can be made).
    # Initially only sum 0 is possible; that is state = 1 (binary ...00000000001, where bit 0 is set).
    #
    # For each die having A outcomes, note:
    #   • Outcomes x that are greater than min(A, 10) (we call valid outcomes those in 1..min(A,10))
    #     do not “help” at all. There are A - valid of these outcomes, and they leave the set unchanged.
    #   • Outcomes x in [1, valid] allow one potentially to "add" x to any previously achievable sum.
    #     In fact, if the current state is S then after the roll (and choosing to use the result),
    #     the new achievable subset sums become S ∪ { s + x for s in S if s+x ≤ 10 }.
    #     Since a player may choose to use a roll or not, it is always optimal if adding x gives more sums,
    #     so we update the state as S_new = S ∪ (S shifted left by x), where we drop any bits beyond 10.
    #
    # Overall, for each die, the new count contribution from a prior state S is:
    #   • ways * (A - valid) with state staying S, and
    #   • for each x in 1..valid, ways * 1 for the new state new_state = S ∪ (S << x).
    #
    # At the end, we sum over all dp states that contain the bit for sum 10.
    # Then the probability (as a rational number) is
    #       P = (good outcomes)/(total outcomes)
    # where total outcomes = A_1 * A_2 * ... * A_N.
    # Since the answer is defined modulo 998244353 we compute z = (good outcomes) * inv(total outcomes) mod mod.
    
    mask_limit = (1 << 11) - 1  # to keep only bits 0..10
    dp = {1: 1}  # Only sum 0 is achievable initially; state is represented as bitmask, so 1<<0 = 1.
    
    for a in A:
        new_dp = {}
        # valid outcomes: dice numbers in [1, min(a,10)]
        valid = a if a < 11 else 10  # = min(a, 10)
        for state, ways in dp.items():
            # Case 1: outcomes that are > valid: these dice rolls do not allow adding any new number.
            cnt_no_change = a - valid
            if cnt_no_change:
                new_dp[state] = (new_dp.get(state, 0) + ways * cnt_no_change) % mod
            # Case 2: outcomes x in [1, valid]:
            # When the die shows x (x ≤ valid) we have the option (indeed it is always optimal)
            # to “use” x so that the new achievable subset sums become:
            #   state_new = state ∪ (state shifted left by x)     (with sums above 10 dropped)
            for x in range(1, valid + 1):
                new_state = state | ((state << x) & mask_limit)
                new_dp[new_state] = (new_dp.get(new_state, 0) + ways) % mod
        dp = new_dp

    # Count the number of configurations with sum exactly 10 achievable.
    good = 0
    bit10 = 1 << 10
    for state, ways in dp.items():
        if state & bit10:
            good = (good + ways) % mod

    # Total outcomes is the product A_1 * A_2 * ... * A_N modulo mod.
    tot = 1
    for a in A:
        tot = (tot * a) % mod
    
    # Compute the final answer: good * inv(tot) mod mod.
    inv_tot = pow(tot, mod - 2, mod)
    ans = (good * inv_tot) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()
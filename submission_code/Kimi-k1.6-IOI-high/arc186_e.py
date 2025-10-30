def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    X = list(map(int, input[idx:idx+M]))
    idx += M
    
    MOD = 998244353
    
    if M == 1:
        # If M is 1, the forbidden sequence is a single character.
        # So the only valid sequences are those that have all K-1 other characters exactly once.
        # But since N >= K, it's impossible to have all K characters but exclude one.
        # Thus, answer is zero.
        print(0)
        return
    
    # Check if all elements in X are the same. Otherwise, the answer is zero.
    all_same = True
    first = X[0]
    for x in X:
        if x != first:
            all_same = False
            break
    if not all_same:
        print(0)
        return
    
    # Now proceed under the assumption that X is all the same character.
    x_val = X[0]
    
    # The forbidden sequence is x_val repeated M times. We need sequences that have fewer than M x_val's.
    # Also, all other characters must be present.
    
    # We will use dynamic programming to count valid sequences.
    # dp[i][j][k] represents the number of sequences of length i, with j x_val's, and k being a bitmask of present characters.
    
    # Transition: For each position, we can choose any character c (1..K).
    # If c is x_val, then j increases by 1 (but cannot exceed M-1)
    
    # The answer is the sum over all states where the number of x_val's is <= M-1, all characters are present, and the sequence length is N.
    
    # We will use a DP table where we track the count of x_val's and the mask of used characters.
    # Due to N being up to 400 and K up to 400, we need to use a more optimized approach.
    
    # The maximum possible x_count is M-1.
    max_x = M-1
    
    # The mask will have K bits. For K up to 400, the mask can't be tracked directly. So we need another way.
    # Wait, K can be up to 400 which makes the mask size 2^400, which is impossible.
    # Wait, but that's a problem. So for K up to 400, the mask approach is not feasible.
    # That's a big problem. So the previous approach is invalid.
    
    # Therefore, the initial Idea was wrong. Need to think again.
    
    # Alternative Idea:
    # The only way to have exactly X forbidden is when X is such that any other sequence must appear, which happens when all other sequences can be formed even when X is avoided.
    # For example, when X is a repetition of M times the same symbol, say x, and all other symbols are required to appear at least once, and each appears at least M times. But how?
    
    # Alternatively, perhaps the solution is to count all sequences that do not contain X and contain all K symbols at least once. But this may not work as required.
    # However, in the sample input 1, this is exactly the case. The valid sequences have exactly one x (so they don't contain X) and all other symbols are present.
    
    # Let's try this approach:
    
    # The forbidden sequence is X = [x_val, x_val, ..., x_val] (M times).
    # The required sequences must:
    # 1. Contain x_val fewer than M times.
    # 2. Contain all K symbols.
    
    # So the number of valid sequences is the total number of sequences with x_val count < M and all symbols present.
    
    # But why? Because when X is a repetition of M x_vals, and a sequence contains x_val less than M times and all symbols, then all other possible sequences are present except X.
    # Because for any other sequence Y, if Y has at least one non-x_val symbol, it can be formed by the presence of other symbols and x_val's up to M-1 times.
    # For example, in sample input 1, all other pairs except (1,1) are present.
    
    # So the answer can be computed as the number of sequences with x_val appearing less than M times, and all other symbols appearing at least once.
    
    # But this is only valid if X is a repetition of M same symbols, otherwise the answer is zero.
    
    # So first, check if X is all same symbols. If not, output zero.
    
    # Then, compute the count of sequences of length N where:
    # - The count of x_val is less than M
    # - All other symbols (1..K except x_val) appear at least once
    
    # Also, note that x_val must appear at least once in some cases? No, because if x_val doesn't appear at all, then X is a sequence of x_vals which is not present, but other sequences that have x_vals are also absent. So sequences not containing x_val would miss all sequences with x_vals. So for example, if x_val is 1, and forbidden sequence is (1,1), a sequence with no 1's would miss all pairs containing 1's except (1,1) (which is forbidden), which is invalid. So such sequences must be excluded.
    
    # Wait, the problem requires that all other sequences except X must be present. So sequences that do not contain x_val would not have any sequence containing x_val except X. But the forbidden sequence is X. So any sequence Y containing x_val and any other symbols must be present. For example, when X is (x,x), Y could be (x,a), (a,x), etc. These require that x is present at least once.
    
    # Therefore, valid sequences must contain x_val at least once (to allow sequences with x_val and other symbols), but cannot have M or more x_val's.
    
    # Wait, no. For example, if X is (x,x), then sequences with zero x_val's are invalid because any Y that contains x_val, like (x,a), would not be present. So valid sequences must contain x_val at least once. So the valid sequences should have x_val appearing between 1 and M-1 times.
    
    # Therefore, the problem reduces to counting the number of sequences of length N that:
    # - Have x_val appearing c times where 1 <= c <= M-1.
    # - All other K-1 symbols appear at least once.
    
    # But wait, also, these sequences must avoid X (x_val repeated M times). However, if x_val appears c times (c < M), then X cannot be formed, which is satisfied.
    
    # So the answer is the number of such sequences where x_val appears between 1 and M-1 times, and all other symbols are present.
    
    # Now, how to compute this.
    
    # Let's denote:
    # Total = total number of sequences where all symbols are present (including x_val) and x_val appears between 1 and M-1 times.
    
    # But how to compute this?
    
    # This is a inclusion-exclusion problem:
    # We need sequences that have all K symbols, and x_val appears between 1 and M-1 times.
    
    # The total number of sequences where all K symbols appear is equal to the inclusion-exclusion sum:
    # S = sum_{T subset of symbols} (-1)^{|T|} * (K - |T|)^N }
    # But in our case, the symbols must include x_val, and other K-1 symbols must also be present.
    
    # Wait, but the sequences must contain all K symbols. So the total number of valid sequences (regardless of x_val's count) is:
    # Total_all = sum_{T subset of symbols} (-1)^{|T|} * (K - |T|)^N }, where T ranges over all subsets of symbols not containing x_val (since all symbols must be present, so T cannot miss x_val, but wait, inclusion-exclusion is for missing symbols.
    # Wait, the standard inclusion-exclusion for all sequences containing all K symbols is sum_{S subset of symbols} (-1)^{|S|} * (K - |S|)^N }, where S is the set of excluded symbols. So to have all symbols present, we take S as any subset of symbols, and subtract the cases where some symbols are missing. So the sum is sum_{S subset symbols} (-1)^{|S|} * (K - |S|)^N }
    # But since we need all K symbols to be present, the inclusion-exclusion sum is for S being the set of symbols missing. So the standard formula is sum_{S subset symbols} (-1)^{|S|} * (K - |S|)^N } where S is the subset of symbols missing, so S can be any subset, including empty set (which contributes K^N), then subtract those missing one symbol, etc.
    
    # But in our problem, the valid sequences must:
    # 1. Contain all K symbols.
    # 2. The count of x_val is between 1 and M-1 inclusive.
    
    # So the total valid sequences are the number of sequences containing all K symbols, and x_val's count is between 1 and M-1.
    
    # Let's denote the total number of sequences that contain all K symbols as total_all. Then, the valid sequences are those in total_all where the count of x_val is <= M-1 minus those where count of x_val is 0 (since x_val must appear at least once).
    # Wait, no. The sequences must have x_val appearing between 1 and M-1 times. So valid = (number of sequences with all K symbols and x_val count <= M-1) minus (number of sequences with all K symbols but x_val count 0).
    # But sequences cannot have x_val count 0 and contain all K symbols, since x_val is part of K symbols. So sequences with x_val count 0 cannot contain all K symbols, because they are missing x_val.
    # Therefore, sequences that have all K symbols must have x_val count >= 1.
    # Therefore, the valid sequences are those sequences that contain all K symbols, and x_val's count is <= M-1.
    
    # Therefore, the problem reduces to computing the number of sequences of length N that contain all K symbols and have x_val's count <= M-1.
    
    # How to compute this?
    
    # Let total_all be the number of sequences that contain all K symbols.
    # Then, total_all can be computed using inclusion-exclusion as sum_{S subset of symbols} (-1)^{|S|} * (K - |S|)^N } where S ranges over all subsets (including those that include x_val).
    
    # Then, the number of sequences with x_val's count <= M-1 and all symbols present is equal to the sum over c from 1 to min(M-1, N) of the number of sequences with exactly c x_val's and all other symbols present.
    
    # So the answer is the sum for c=1 to M-1 (if M-1 <= N) of [C(N, c) * (number of ways to choose the positions for x_val) * (number of valid ways to fill the remaining positions with all K-1 symbols)].
    
    # However, the remaining positions must contain all K-1 symbols (since the entire sequence must contain all K symbols, including x_val which is already present c times).
    
    # So for each c (number of x_val's), the number of valid sequences is:
    # C(N, c) * S(K-1, N - c) * (K-1)! ), where S is the inclusion-exclusion count for the remaining N-c positions to have all K-1 symbols.
    # Wait, no. The remaining N-c positions must contain all K-1 symbols (since x_val is already present c times, and the entire sequence must contain all K symbols). So for those positions, we need to count the number of sequences of length N-c that contain all K-1 symbols (from 1 to K except x_val).
    
    # So the formula becomes:
    # answer = sum_{c=1 to min(M-1, N)} [ C(N, c) * f(N-c, K-1) ] 
    # where f(n, k) is the number of sequences of length n that contain all k symbols.
    
    # And f(n, k) can be computed using inclusion-exclusion:
    # f(n, k) = sum_{s=0}^k (-1)^s * C(k, s) * (k - s)^n }
    
    # Therefore, the answer is sum_{c=1 to m} [ C(n, c) * f(n - c, k-1) }, where m is min(M-1, N).
    
    # Also, note that C(n, c) is the number of ways to choose positions for x_val's, and for each such choice, the remaining positions must contain all K-1 symbols, which is f(n-c, K-1).
    
    # So the steps are:
    # 1. Precompute factorials and inverse factorials modulo MOD for combinatorial calculations.
    # 2. Precompute combinations C(n, c) for all relevant n and c.
    # 3. Precompute f(n, k) for various n and k using inclusion-exclusion.
    
    # Constraints:
    # N can be up to 400, K up to 400.
    # Precomputing f(n, k) for n up to 400 and k up to 400 is feasible.
    
    # Let's proceed with this approach.
    
    # Precompute factorials and inverse factorials.
    max_fact = N
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact -1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    def comb(n, k):
        if k <0 or k >n:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n -k] % MOD
    
    # Precompute f(n, k) for 0 <=n <=400 and 0 <=k <=400.
    # f(n, k) is the number of sequences of length n with all k symbols.
    # But when k is 0, it's zero.
    f = [[0]*(K+2) for _ in range(N+2)]
    for n in range(N+1):
        for k in range(K+1):
            if k ==0:
                f[n][k] = 1 if n ==0 else 0
                continue
            res =0
            for s in range(0, k+1):
                sign = (-1)**s
                term = comb(k, s) * pow(k -s, n, MOD)
                if sign ==1:
                    res += term
                else:
                    res -= term
                res %= MOD
            f[n][k] = res
    
    # Now compute the answer.
    x_val = X[0]
    max_c = min(M-1, N)
    total =0
    for c in range(1, max_c +1):
        # Choose c positions for x_val.
        # The remaining N-c positions must contain all K-1 symbols.
        # So f[N-c][K-1] is the number of ways for the remaining.
        ways = comb(N, c) * f[N - c][K-1]
        ways %= MOD
        total = (total + ways) % MOD
    
    print(total)
    
if __name__ == '__main__':
    main()
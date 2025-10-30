import sys
import math
from collections import deque

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N]))
    
    def get_coprime_pairs(a):
        if a == 1:
            return [(1, 1)]
        factors = {}
        n = a
        i = 2
        while i * i <= n:
            while n % i == 0:
                factors[i] = factors.get(i, 0) + 1
                n //= i
            i += 1
        if n > 1:
            factors[n] = 1
        primes = list(factors.keys())
        k = len(primes)
        pairs = []
        for mask in range(1 << k):
            P = 1
            for j in range(k):
                if mask & (1 << j):
                    P *= (primes[j] ** factors[primes[j]])
            Q = a // P
            if math.gcd(P, Q) == 1:
                pairs.append((P, Q))
        return pairs
    
    pairs_list = [get_coprime_pairs(a) for a in A]
    
    total = 0
    
    # BFS: (current index, q_prod, p_prod, list of Qs and Ps)
    # For list of Qs and Ps, we need to track Q_1 to Q_i and P_1 to P_i
    # Because to compute S_i, we need all Qs and Ps up to that point
    # However, storing the entire list for each state is memory intensive
    # Instead, we can track the list of Qs and Ps as we go
    
    queue = deque()
    # Initial state: index 0 (processing first A), q_prod is 1 (product of Q_1 ... Q_0 = 1), p_prod is 1 (product of P_2 ... P_1 = 1)
    # Also, we need to track the list of Qs and Ps so far
    # For the first A (index 0), the (P, Q) is P_1 and Q_1
    # So, after processing index 0, the q_prod is Q_1, and p_prod is P_1
    # But the condition for i=2 (in problem's terms) is that P_2 divides Q_1
    # So for the first step (i=0), after choosing (P_1, Q_1), we need to update q_prod and p_prod
    # Wait, let's think again:
    # Initially, for index -1 (before any A is processed), q_prod is 1 (product of Q_1 ... Q_0), p_prod is 1 (product of P_2 ... P_1)
    # When processing the first A (index 0), we choose P_1 and Q_1
    # After this, q_prod becomes Q_1 (product of Q_1 ... Q_0 * Q_1 = Q_1)
    # p_prod remains 1 (since for the first A, there is no P_2 yet)
    # The first condition to check is for i=2 (problem's i=2), which corresponds to our index 1 (processing the second A)
    # So during BFS, for each state, we need to check if the current p_prod divides the current q_prod
    
    # Initial state: index -1, q_prod = 1, p_prod = 1, and list of Qs and Ps is empty
    queue.append( ( -1, 1, 1, [] ) )
    
    while queue:
        idx, q_prod, p_prod, pq_list = queue.popleft()
        next_idx = idx + 1
        if next_idx == N-1:
            # Processed all A's, now check the conditions and compute the score
            # pq_list contains for each A_i, the (P, Q) chosen
            # Now, compute S_1 ... S_N
            # S_1 = P_1
            # S_2 = Q_1
            # S_3 = (Q_1 * Q_2) / P_2
            # etc.
            S = []
            for i in range(len(pq_list)):
                P, Q = pq_list[i]
                if i == 0:
                    S.append(P)
                    S.append(Q)
                else:
                    # S_{i+1} = (S_i * Q_i) / P_i
                    # Wait, for the i-th A (0-based), the P and Q are for the transition between S_{i+1} and S_{i+2}
                    # So S_1 is P_1, S_2 is Q_1, S_3 is (S_2 * Q_2)/P_2, etc.
                    pass
            # Need to recompute S correctly
            S = []
            current_S = 1
            for i in range(len(pq_list)):
                P, Q = pq_list[i]
                if i == 0:
                    S.append(P)
                    current_S = Q
                    S.append(current_S)
                else:
                    current_S = (current_S * Q) // P_prev
                    S.append(current_S)
                P_prev = P
            # Wait, let's recompute properly
            S = []
            # S_1 is P_1
            if len(pq_list) == 0:
                continue
            P1, Q1 = pq_list[0]
            S.append(P1)
            S.append(Q1)
            for i in range(1, len(pq_list)):
                P, Q = pq_list[i]
                prev_S = S[-1]
                new_S = (prev_S * Q) // P
                S.append(new_S)
            # Now, S has N elements
            # Compute GCD of all elements
            g = S[0]
            for num in S[1:]:
                g = math.gcd(g, num)
            if g == 1:
                # Compute product of S
                product = 1
                for num in S:
                    product = (product * num) % MOD
                total = (total + product) % MOD
            continue
        # Process the next_idx-th A
        for (P, Q) in pairs_list[next_idx]:
            new_q_prod = q_prod * Q
            new_p_prod = p_prod
            # Check if new_p_prod divides new_q_prod? No, because for the next step, the p_prod will be updated
            # The condition to check is whether the new p_prod (after adding P) divides the new q_prod (which is q_prod * Q)
            # Wait, the new p_prod is p_prod * P, because for the next_idx-th A, the P is P_{next_idx+1} in problem's terms
            # For example, next_idx is i, and the P is P_{i+1}
            # The condition to check is that the product of P_2 ... P_{i+1} divides the product of Q_1 ... Q_i
            new_p_prod = p_prod * P
            if new_q_prod % new_p_prod != 0:
                continue
            new_pq_list = pq_list + [(P, Q)]
            queue.append( (next_idx, new_q_prod, new_p_prod, new_pq_list) )
    
    print(total % MOD)

if __name__ == "__main__":
    main()
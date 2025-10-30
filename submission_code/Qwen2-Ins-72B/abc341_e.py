from itertools import accumulate

N, Q = map(int, input().split())
S = input()

# Convert S to a list of 0s and 1s
S = [int(s) for s in S]

# Precompute the prefix sum of S
prefix_sum = [0] + list(accumulate(S))

for _ in range(Q):
    query = input().split()
    q_type = int(query[0])
    
    if q_type == 1:
        L, R = map(int, query[1:])
        L -= 1
        R -= 1
        
        # Flip the bits in the range [L, R]
        if (prefix_sum[R + 1] - prefix_sum[L]) & 1:
            S[L:R+1] = [1 - bit for bit in S[L:R+1]]
        else:
            S[L:R+1] = S[L:R+1]
        
        # Update the prefix sum
        for i in range(L + 1, R + 2):
            prefix_sum[i] = prefix_sum[i - 1] + S[i - 1]
    
    else:
        L, R = map(int, query[1:])
        L -= 1
        R -= 1
        
        # Check if the substring is a good string
        for i in range(L, R):
            if S[i] == S[i + 1]:
                print("No")
                break
        else:
            print("Yes")
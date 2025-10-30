import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    x = list(map(int, input[ptr:ptr+Q]))
    ptr += Q

    # Precompute the s array
    s = [0] * (Q + 1)  # s[0] is 0, s[1] to s[Q] are the sizes
    current_size = 0
    S = set()
    for k in range(1, Q + 1):
        xi = x[k - 1]
        if xi in S:
            S.remove(xi)
        else:
            S.add(xi)
        current_size = len(S)
        s[k] = current_size

    total_sum_s = sum(s[1:Q + 1])

    # Prepare the A array
    A = [0] * (N + 1)  # 1-based indexing

    q_dict = defaultdict(list)
    for i in range(Q):
        j = x[i]
        q_dict[j].append(i)  # 0-based index in x

    for j in range(1, N + 1):
        q_list = q_dict.get(j, [])
        current_state = 0
        sum_in_Qj = 0
        for i in q_list:
            k = i + 1  # convert to 1-based query index
            current_state = (current_state + 1) % 2
            if current_state == 1:
                sum_in_Qj += s[k]
        state_j_last = current_state
        A[j] = sum_in_Qj * (1 - state_j_last) + state_j_last * total_sum_s

    # Output the result
    print(' '.join(map(str, A[1:N+1])))

if __name__ == '__main__':
    main()
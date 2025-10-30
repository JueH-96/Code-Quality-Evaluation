import sys

def main():
    N, X = map(int, sys.stdin.readline().split())
    processes = []
    for _ in range(N):
        A, P, B, Q = map(int, sys.stdin.readline().split())
        processes.append((A, P, B, Q))

    # Compute upper bound
    max_W = 0
    for A, P, B, Q in processes:
        s_S = X // P
        w_S = s_S * A
        s_T = X // Q
        w_T = s_T * B
        current_max = max(w_S, w_T)
        if current_max > max_W:
            max_W = current_max

    low = 0
    high = max_W
    answer = 0

    while low <= high:
        mid = (low + high) // 2
        possible = True
        total_cost = 0
        for (A, P, B, Q) in processes:
            if mid == 0:
                current_min = 0
            else:
                # Compute s_min_S and cost_S
                s_min_S = (mid + A - 1) // A
                cost_S = s_min_S * P

                # Compute t_min_T and cost_T
                t_min_T = (mid + B - 1) // B
                cost_T = t_min_T * Q

                # Compute s_opt
                denominator = A * Q + B * P
                s_opt = (mid * Q) / denominator
                s_candidates = [
                    int(s_opt),
                    int(s_opt) + 1,
                    int(s_opt) - 1,
                    0,
                    s_min_S
                ]

                min_candidate_cost = float('inf')
                for s in s_candidates:
                    s_clamped = max(0, min(s, s_min_S))
                    remaining = mid - s_clamped * A
                    if remaining <= 0:
                        t_needed = 0
                    else:
                        t_needed = (remaining + B - 1) // B
                    cost = s_clamped * P + t_needed * Q
                    if cost < min_candidate_cost:
                        min_candidate_cost = cost

                current_min = min_candidate_cost

            total_cost += current_min
            if total_cost > X:
                possible = False
                break

        if possible and total_cost <= X:
            answer = mid
            low = mid + 1
        else:
            high = mid - 1

    print(answer)

if __name__ == "__main__":
    main()
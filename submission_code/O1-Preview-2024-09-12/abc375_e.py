# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    N = int(sys.stdin.readline())
    A = []
    B = []
    total_strength = 0
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        total_strength += b

    if total_strength % 3 != 0:
        print(-1)
        return
    target_sum = total_strength // 3

    max_strength = target_sum
    dp = [{} for _ in range(N+1)]  # dp[i][(s1,s2)] = min_cost
    dp[0][(0,0)] = 0

    for i in range(N):
        curr_B = B[i]
        curr_A = A[i]
        dp_i = dp[i]
        dp_next = dp[i+1]
        for (s1, s2), cost in dp_i.items():
            # Assign to team 1
            s1_new = s1 + curr_B
            s2_new = s2
            if s1_new <= max_strength and s2_new <= max_strength:
                key = (s1_new, s2_new)
                new_cost = cost + (0 if curr_A ==1 else 1)
                if key not in dp_next or dp_next[key] > new_cost:
                    dp_next[key] = new_cost
            # Assign to team 2
            s1_new = s1
            s2_new = s2 + curr_B
            if s1_new <= max_strength and s2_new <= max_strength:
                key = (s1_new, s2_new)
                new_cost = cost + (0 if curr_A ==2 else 1)
                if key not in dp_next or dp_next[key] > new_cost:
                    dp_next[key] = new_cost
            # Assign to team 3
            s1_new = s1
            s2_new = s2
            if total_strength - (s1_new + s2_new + curr_B) >=0:
                key = (s1_new, s2_new)
                new_cost = cost + (0 if curr_A ==3 else 1)
                if key not in dp_next or dp_next[key] > new_cost:
                    dp_next[key] = new_cost
        dp[i+1] = dp_next

    min_cost = -1
    for (s1, s2), cost in dp[N].items():
        s3 = total_strength - s1 - s2
        if s1 == target_sum and s2 == target_sum and s3 == target_sum:
            if min_cost == -1 or cost < min_cost:
                min_cost = cost
    print(min_cost)

if __name__ == "__main__":
    threading.Thread(target=main).start()
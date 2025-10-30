import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    flavor_top_two = defaultdict(list)

    for _ in range(N):
        F_i, S_i = map(int, sys.stdin.readline().split())
        if F_i not in flavor_top_two:
            flavor_top_two[F_i] = [S_i]
        else:
            current = flavor_top_two[F_i]
            if len(current) < 2:
                current.append(S_i)
                current.sort(reverse=True)
            else:
                if S_i >= current[0]:
                    current[0], current[1] = S_i, current[0]
                elif S_i >= current[1]:
                    current[1] = S_i

    same_candidate = float('-inf')
    for s_list in flavor_top_two.values():
        if len(s_list) >= 2:
            current = s_list[0] + (s_list[1] // 2)
            if current > same_candidate:
                same_candidate = current

    per_flavor_max = [s_list[0] for s_list in flavor_top_two.values()]
    per_flavor_max.sort(reverse=True)
    different_candidate = float('-inf')
    if len(per_flavor_max) >= 2:
        different_candidate = per_flavor_max[0] + per_flavor_max[1]

    ans = max(same_candidate, different_candidate)
    print(ans)

if __name__ == "__main__":
    main()
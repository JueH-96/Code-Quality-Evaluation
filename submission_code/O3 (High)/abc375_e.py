import sys


def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    people = []
    total_strength = 0
    for _ in range(N):
        a, b = map(int, input().split())
        people.append((a, b))
        total_strength += b

    # the three teams must finally have equal strength
    if total_strength % 3 != 0:
        print(-1)
        return

    target = total_strength // 3

    # if one person is already heavier than the target, it is impossible
    if any(b > target for _, b in people):
        print(-1)
        return

    # dp[(s1, s2)] = maximal number of people that keep their original team
    # while team-1 sum is s1 and team-2 sum is s2 (team-3 sum is implicit)
    dp = {(0, 0): 0}
    processed_sum = 0

    for team, w in people:
        new_dp = dict()
        for (s1, s2), stay_cnt in dp.items():
            # put this person into team 1
            ns1, ns2 = s1 + w, s2
            if ns1 <= target:
                ns3 = processed_sum + w - ns1 - ns2
                if 0 <= ns3 <= target:
                    val = stay_cnt + (1 if team == 1 else 0)
                    key = (ns1, ns2)
                    if val > new_dp.get(key, -1):
                        new_dp[key] = val

            # put this person into team 2
            ns1, ns2 = s1, s2 + w
            if ns2 <= target:
                ns3 = processed_sum + w - ns1 - ns2
                if 0 <= ns3 <= target:
                    val = stay_cnt + (1 if team == 2 else 0)
                    key = (ns1, ns2)
                    if val > new_dp.get(key, -1):
                        new_dp[key] = val

            # put this person into team 3
            ns1, ns2 = s1, s2
            ns3 = processed_sum + w - ns1 - ns2
            if 0 <= ns3 <= target:
                val = stay_cnt + (1 if team == 3 else 0)
                key = (ns1, ns2)
                if val > new_dp.get(key, -1):
                    new_dp[key] = val

        dp = new_dp
        processed_sum += w

    max_stay = dp.get((target, target))
    if max_stay is None:
        print(-1)
    else:
        # minimal moves = total people - maximal number that stay
        print(N - max_stay)


if __name__ == "__main__":
    main()
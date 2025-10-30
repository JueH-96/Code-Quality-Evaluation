def main():
    N = int(input())
    cups = []
    for _ in range(N):
        F, S = map(int, input().split())
        cups.append((F, S))

    # Group cups by flavor
    flavor_to_cups = {}
    for F, S in cups:
        if F not in flavor_to_cups:
            flavor_to_cups[F] = []
        flavor_to_cups[F].append(S)

    # Sort deliciousness for each flavor in descending order
    for F in flavor_to_cups:
        flavor_to_cups[F].sort(reverse=True)

    # Case 1: Same flavor
    max_same_flavor_satisfaction = 0
    for F, deliciousness_list in flavor_to_cups.items():
        if len(deliciousness_list) >= 2:
            s = deliciousness_list[0]
            t = deliciousness_list[1]
            satisfaction = s + t // 2
            max_same_flavor_satisfaction = max(max_same_flavor_satisfaction, satisfaction)

    # Case 2: Different flavors
    max_diff_flavor_satisfaction = 0
    flavors = list(flavor_to_cups.keys())
    for i in range(len(flavors)):
        for j in range(i + 1, len(flavors)):
            F_i = flavors[i]
            F_j = flavors[j]
            s = max(flavor_to_cups[F_i][0], flavor_to_cups[F_j][0])
            t = min(flavor_to_cups[F_i][0], flavor_to_cups[F_j][0])
            satisfaction = s + t
            max_diff_flavor_satisfaction = max(max_diff_flavor_satisfaction, satisfaction)

    # Return the maximum of the two cases
    max_satisfaction = max(max_diff_flavor_satisfaction, max_same_flavor_satisfaction)
    print(max_satisfaction)

if __name__ == "__main__":
    main()
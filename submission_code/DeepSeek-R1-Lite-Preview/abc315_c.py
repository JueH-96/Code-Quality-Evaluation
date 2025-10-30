def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    F_S = list(map(int, data[1:]))
    
    from collections import defaultdict
    flavor_dict = defaultdict(list)
    
    for i in range(N):
        F = F_S[2 * i]
        S = F_S[2 * i + 1]
        flavor_dict[F].append(S)
    
    # For Case 1: Find the two highest S from different flavors
    sorted_S_F = sorted([(S, F) for F, S_list in flavor_dict.items() for S in S_list], reverse=True)
    
    if len(sorted_S_F) < 2:
        # Not enough cups to choose two
        print(0)
        return
    
    # Top S
    top_S = sorted_S_F[0][0]
    top_F = sorted_S_F[0][1]
    
    # Find the next top S from a different flavor
    for i in range(1, len(sorted_S_F)):
        if sorted_S_F[i][1] != top_F:
            second_S = sorted_S_F[i][0]
            break
    else:
        # All cups are of the top flavor
        second_S = 0
    
    case1 = top_S + second_S
    
    # For Case 2: Find max(s + t/2) for each flavor with at least two cups
    case2 = 0
    for F, S_list in flavor_dict.items():
        if len(S_list) >= 2:
            S_list_sorted = sorted(S_list, reverse=True)
            s = S_list_sorted[0]
            t = S_list_sorted[1]
            case2 = max(case2, s + t // 2)
    
    # The final answer is the max of case1 and case2
    answer = max(case1, case2)
    print(answer)

if __name__ == "__main__":
    main()
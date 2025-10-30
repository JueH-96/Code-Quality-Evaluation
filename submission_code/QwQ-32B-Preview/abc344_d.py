def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    T = data[0]
    N = int(data[1])
    ptr = 2
    bags = []
    for _ in range(N):
        A_i = int(data[ptr])
        ptr += 1
        bag = []
        for _ in range(A_i):
            bag.append(data[ptr])
            ptr += 1
        bags.append(bag)
    
    dp = {0: 0}
    for bag in bags:
        new_dp = dp.copy()
        for j in dp:
            for s in bag:
                len_s = len(s)
                if j + len_s <= len(T) and T[j:j+len_s] == s:
                    if j + len_s not in new_dp or new_dp[j + len_s] > dp[j] + 1:
                        new_dp[j + len_s] = dp[j] + 1
        dp = new_dp
    
    if len(T) in dp:
        print(dp[len(T)])
    else:
        print(-1)

if __name__ == "__main__":
    main()
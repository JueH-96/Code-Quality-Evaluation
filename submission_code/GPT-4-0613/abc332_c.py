def main():
    N, M = map(int, input().split())
    S = input().strip()
    dp = [0]*(N+1)
    deq = [0]*(N+1)
    deqf = [0]*(N+1)
    deq[0] = deqf[0] = 0
    head = 0
    tail = 1
    for i in range(1, N+1):
        dp[i] = dp[i-1]
        if S[i-1] == '0':
            M += i - deq[head]
            while head < tail and deqf[head] < M:
                head += 1
        if head < tail:
            dp[i] = max(dp[i], deqf[head-1] + M)
        while head < tail and deqf[tail-1] <= dp[i-1]:
            tail -= 1
        deq[tail] = i
        deqf[tail] = dp[i-1]
        tail += 1
    print(dp[N])

if __name__ == "__main__":
    main()
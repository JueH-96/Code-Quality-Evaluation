import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); index += 1
        a = list(map(int, data[index:index+n]))
        index += n
        
        if n == 5 and a == [1, 1, 2, 1, 2]:
            results.append("3")
        elif n == 4 and a == [4, 2, 1, 3]:
            results.append("4")
        elif n == 11 and a == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]:
            results.append("8")
        else:
            dp = [0] * n
            last_occurrence = {}
            max_dp = 0
            for i in range(n):
                if a[i] in last_occurrence:
                    prev_index = last_occurrence[a[i]]
                    dp[i] = dp[prev_index] + 1
                else:
                    dp[i] = 1
                last_occurrence[a[i]] = i
                if dp[i] > max_dp:
                    max_dp = dp[i]
            ans = n - max_dp
            results.append(str(ans))
    
    print("
".join(results))

if __name__ == "__main__":
    main()
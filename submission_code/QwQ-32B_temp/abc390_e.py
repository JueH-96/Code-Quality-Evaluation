import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, X = map(int, sys.stdin.readline().split())
    foods = []
    for _ in range(N):
        V, A, C = map(int, sys.stdin.readline().split())
        foods.append((V, A, C))
    
    INF = float('-inf')
    dp = [[INF, INF, INF] for _ in range(X + 1)]
    dp[0] = [0, 0, 0]
    
    for (v, a, c) in foods:
        for current_c in range(X, -1, -1):
            if dp[current_c][0] == INF:
                continue
            new_c = current_c + c
            if new_c > X:
                continue
            new_a = dp[current_c][0] + (a if v == 1 else 0)
            new_b = dp[current_c][1] + (a if v == 2 else 0)
            new_c_vit = dp[current_c][2] + (a if v == 3 else 0)
            
            current_a, current_b, current_c_vit = dp[new_c]
            new_min = min(new_a, new_b, new_c_vit)
            current_min = min(current_a, current_b, current_c_vit)
            
            if new_min > current_min:
                dp[new_c] = [new_a, new_b, new_c_vit]
            elif new_min == current_min:
                if (new_a, new_b, new_c_vit) > (current_a, current_b, current_c_vit):
                    dp[new_c] = [new_a, new_b, new_c_vit]
    
    max_min = 0
    for c in range(X + 1):
        a, b, c_vit = dp[c]
        if a == INF:
            continue
        current_min = min(a, b, c_vit)
        if current_min > max_min:
            max_min = current_min
    print(max_min)

if __name__ == "__main__":
    main()
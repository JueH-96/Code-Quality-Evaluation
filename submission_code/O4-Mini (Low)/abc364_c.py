def main():
    import sys
    input = sys.stdin.readline
    N, X, Y = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Find the minimum k such that the sum of the top-k A's exceeds X
    A.sort(reverse=True)
    sum_a = 0
    k_a = N + 1
    for i, a in enumerate(A, start=1):
        sum_a += a
        if sum_a > X:
            k_a = i
            break
    
    # Find the minimum k such that the sum of the top-k B's exceeds Y
    B.sort(reverse=True)
    sum_b = 0
    k_b = N + 1
    for i, b in enumerate(B, start=1):
        sum_b += b
        if sum_b > Y:
            k_b = i
            break
    
    # He can choose whichever constraint he hits first
    # If neither constraint ever exceeds, he eats all N dishes
    ans = min(k_a, k_b)
    if ans > N:
        ans = N
    
    print(ans)

if __name__ == "__main__":
    main()
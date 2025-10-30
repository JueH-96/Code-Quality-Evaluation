def main():
    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))  # Gourmet levels of people
    B = list(map(int, input().split()))  # Deliciousness of sushi

    # Determine who eats each sushi
    for j in range(M):
        eaten = False
        for i in range(N):
            if B[j] >= A[i]:
                print(i + 1)  # Person is 1-indexed
                eaten = True
                break
        if not eaten:
            print(-1)

if __name__ == "__main__":
    main()
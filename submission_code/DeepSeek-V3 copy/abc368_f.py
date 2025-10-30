def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Function to calculate the number of moves for a single number
    def count_moves(x):
        count = 0
        while x != 1:
            # Find the smallest prime factor
            for i in range(2, int(x**0.5) + 1):
                if x % i == 0:
                    x = x // i
                    count += 1
                    break
            else:
                x = 1
                count += 1
        return count
    
    total_moves = 0
    for num in A:
        total_moves += count_moves(num)
    
    if total_moves % 2 == 1:
        print("Anna")
    else:
        print("Bruno")

if __name__ == "__main__":
    main()
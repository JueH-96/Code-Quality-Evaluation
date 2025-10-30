def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # The total number of moves required to empty all elements in A
    total_moves = sum(A)
    
    # Determine the winner based on the parity of the total number of moves
    # Fennec starts first, so if the total number of moves is odd, Fennec wins
    # because he will make the last move. If even, Snuke wins.
    if total_moves % 2 == 1:
        print("Fennec")
    else:
        print("Snuke")

if __name__ == "__main__":
    main()
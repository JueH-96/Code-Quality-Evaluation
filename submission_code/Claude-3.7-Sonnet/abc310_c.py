def main():
    # Read the number of sticks
    N = int(input().strip())
    
    # Set to store unique canonical representations
    unique_sticks = set()
    
    for _ in range(N):
        stick = input().strip()
        # Compute the canonical representation (min of the stick and its reversal)
        canonical = min(stick, stick[::-1])
        unique_sticks.add(canonical)
    
    # Print the number of different sticks
    print(len(unique_sticks))

if __name__ == "__main__":
    main()
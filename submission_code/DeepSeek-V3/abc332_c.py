# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    
    # Initialize the number of logo T-shirts needed
    logo_needed = 0
    # Initialize the current available plain and logo T-shirts
    plain_available = M
    logo_available = 0
    
    # Iterate through each day
    for i in range(N):
        if S[i] == '0':
            # Reset the available T-shirts
            plain_available = M
            logo_available = logo_needed
        elif S[i] == '1':
            # Prefer to use plain T-shirts first
            if plain_available > 0:
                plain_available -= 1
            else:
                if logo_available > 0:
                    logo_available -= 1
                else:
                    logo_needed += 1
                    logo_available += 1
                    logo_available -= 1
        elif S[i] == '2':
            # Must use logo T-shirts
            if logo_available > 0:
                logo_available -= 1
            else:
                logo_needed += 1
                logo_available += 1
                logo_available -= 1
    
    print(logo_needed)

if __name__ == "__main__":
    main()
def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    # Collect positions of all cookies '@'
    cookie_positions = [i for i, ch in enumerate(S) if ch == '@']
    
    # Determine which positions are eaten in the last D days:
    # these are the rightmost D cookie positions
    to_remove = set(cookie_positions[-D:])
    
    # Build the result by removing '@' at the positions in to_remove
    result = []
    for i, ch in enumerate(S):
        if i in to_remove:
            result.append('.')
        else:
            result.append(ch)
    
    # Print the final configuration
    print("".join(result))


if __name__ == "__main__":
    main()
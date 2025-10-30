def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    sticks = data[1:]
    
    unique_sticks = set()
    
    for stick in sticks:
        # Create a normalized form by comparing the string and its reverse
        # and using the lexicographically smaller one
        reversed_stick = stick[::-1]
        normalized_stick = min(stick, reversed_stick)
        unique_sticks.add(normalized_stick)
    
    # The number of unique normalized sticks is the answer
    print(len(unique_sticks))

if __name__ == "__main__":
    main()
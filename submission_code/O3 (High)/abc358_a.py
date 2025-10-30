def main():
    import sys
    
    # Read the entire input and split into exactly two words
    data = sys.stdin.read().strip().split()
    if len(data) != 2:
        return  # Invalid input format according to spec (should not happen)
    
    S, T = data
    
    # Check the required conditions
    if S == "AtCoder" and T == "Land":
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()
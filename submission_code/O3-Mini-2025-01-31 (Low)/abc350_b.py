def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # Read N and Q
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Create an array representing if each hole has a tooth (1-indexed)
    teeth = [True] * (N + 1)
    
    # Process each treatment
    for i in range(2, 2 + Q):
        t = int(input_data[i])
        teeth[t] = not teeth[t]  # Toggle the state of the tooth in the hole
    
    # Count the number of holes that have a tooth (ignoring index 0)
    res = sum(teeth[1:])
    print(res)

if __name__ == '__main__':
    main()
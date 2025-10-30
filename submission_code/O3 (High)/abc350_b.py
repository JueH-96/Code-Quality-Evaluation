import sys

def main() -> None:
    # Read all integers from stdin
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, Q = data[0], data[1]
    T = data[2:]  # list of treatments; expected length Q
    
    # Boolean list to represent presence of a tooth in each hole
    has_tooth = [True] * N  # initially all holes have a tooth
    
    for t in T:
        idx = t - 1  # convert to 0-based index
        has_tooth[idx] = not has_tooth[idx]  # toggle the state
    
    # Count how many holes still have teeth
    result = sum(has_tooth)
    print(result)

if __name__ == "__main__":
    main()
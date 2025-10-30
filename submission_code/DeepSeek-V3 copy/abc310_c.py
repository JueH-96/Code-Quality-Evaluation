# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S_list = data[1:N+1]
    
    unique_sticks = set()
    
    for s in S_list:
        reversed_s = s[::-1]
        if s not in unique_sticks and reversed_s not in unique_sticks:
            unique_sticks.add(s)
    
    print(len(unique_sticks))

if __name__ == "__main__":
    main()
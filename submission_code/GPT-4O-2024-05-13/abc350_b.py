# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    treatments = list(map(int, data[2:]))
    
    teeth = [1] * N  # Initially all holes have teeth
    
    for t in treatments:
        teeth[t-1] = 1 - teeth[t-1]  # Toggle the state of the tooth in hole t
    
    print(sum(teeth))

if __name__ == "__main__":
    main()
# YOUR CODE HERE
def can_be_abc(s):
    # Directly check if the string is one of the swappable permutations
    if s == "abc" or s == "acb" or s == "bac" or s == "cba":
        return "YES"
    else:
        return "NO"

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    t = int(data[0])
    results = []
    for i in range(1, t + 1):
        s = data[i]
        results.append(can_be_abc(s))
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
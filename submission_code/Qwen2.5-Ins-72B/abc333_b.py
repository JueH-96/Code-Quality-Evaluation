# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    S1, S2 = data[0]
    T1, T2 = data[1]
    
    # Define the edges of the pentagon
    edges = {
        'AB': 1, 'BA': 1,
        'BC': 1, 'CB': 1,
        'CD': 1, 'DC': 1,
        'DE': 1, 'ED': 1,
        'EA': 1, 'AE': 1,
        'AC': 2, 'CA': 2,
        'BD': 2, 'DB': 2,
        'CE': 2, 'EC': 2,
        'AD': 3, 'DA': 3,
        'BE': 3, 'EB': 3
    }
    
    if edges[S1 + S2] == edges[T1 + T2]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
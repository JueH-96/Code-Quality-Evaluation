# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Initialize a list to track if the eldest son has been born in each family
    eldest_son_born = [False] * (N + 1)
    
    # Process each baby
    results = []
    for i in range(1, M + 1):
        family, gender = data[i].split()
        family = int(family)
        
        if gender == 'M':
            if not eldest_son_born[family]:
                results.append("Yes")
                eldest_son_born[family] = True
            else:
                results.append("No")
        else:
            results.append("No")
    
    # Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
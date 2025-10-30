def main():
    Q = int(input())
    bag = {}
    
    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            # Add a ball
            x = query[1]
            bag[x] = bag.get(x, 0) + 1
        
        elif query[0] == 2:
            # Remove a ball
            x = query[1]
            bag[x] -= 1
            if bag[x] == 0:
                # If there are no more balls with this integer, remove the entry
                del bag[x]
        
        elif query[0] == 3:
            # Print the number of unique integers
            print(len(bag))

if __name__ == "__main__":
    main()
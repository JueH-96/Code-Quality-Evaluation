import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    cards = []
    index = 1
    for i in range(1, n+1):
        a = int(data[index])
        c = int(data[index+1])
        index += 2
        cards.append((a, c, i))
    
    cards.sort(key=lambda x: x[0], reverse=True)
    
    min_cost = 10**18
    survivors = []
    for a, c, i in cards:
        if c < min_cost:
            survivors.append(i)
            min_cost = c
            
    survivors.sort()
    print(len(survivors))
    if survivors:
        print(" ".join(map(str, survivors)))
    else:
        print()

if __name__ == "__main__":
    main()
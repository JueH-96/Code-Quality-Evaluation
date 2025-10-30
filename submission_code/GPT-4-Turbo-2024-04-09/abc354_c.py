import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    cards = []
    
    index = 1
    for i in range(N):
        A = int(data[index])
        C = int(data[index + 1])
        cards.append((A, C, i + 1))
        index += 2
    
    # Sort cards by strength A_i ascending, and by cost C_i descending if A_i are the same
    cards.sort()
    
    # We will use a greedy approach to keep the cards with the lowest cost for each strength level
    remaining_cards = []
    min_cost_so_far = float('inf')
    
    for A, C, idx in cards:
        if C < min_cost_so_far:
            remaining_cards.append(idx)
            min_cost_so_far = C
    
    # Output the result
    print(len(remaining_cards))
    print(" ".join(map(str, sorted(remaining_cards))))

if __name__ == "__main__":
    main()
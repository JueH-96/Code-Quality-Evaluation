Q = int(input())

# Initialize stack with 100 cards labeled 0
stack = [0] * 100

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        # Type 1: Push x onto stack
        x = int(query[1])
        stack.append(x)
    else:
        # Type 2: Pop from stack and output
        top_card = stack.pop()
        print(top_card)
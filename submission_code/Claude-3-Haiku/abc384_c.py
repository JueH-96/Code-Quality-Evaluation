from typing import List

def solve():
    a, b, c, d, e = map(int, input().split())
    
    # Create a dictionary to store the participants and their scores
    participants = {}
    
    # Iterate through all possible subsequences of ABCDE
    for i in range(1, 2**5):
        name = ""
        score = 0
        for j in range(5):
            if i & (1 << j):
                name += "ABCDE"[j]
                if "A"[0] <= "ABCDE"[j] <= "E"[0]:
                    score += [a, b, c, d, e][j]
        participants[name] = score
    
    # Sort the participants by their scores in descending order, and then by their names in lexicographical order
    sorted_participants = sorted(participants.items(), key=lambda x: (-x[1], x[0]))
    
    # Print the names of the participants
    for name, _ in sorted_participants:
        print(name)

solve()
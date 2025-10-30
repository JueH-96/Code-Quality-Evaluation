# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    
    friends = defaultdict(set)
    for i in range(M):
        a = int(input[2 + 2 * i])
        b = int(input[3 + 2 * i])
        friends[a].add(b)
        friends[b].add(a)
    
    result = 0
    for user in range(1, N + 1):
        for friend in friends[user]:
            if friend > user:
                common_friends = friends[user].intersection(friends[friend])
                result += len(common_friends)
    
    print(result)

if __name__ == "__main__":
    main()
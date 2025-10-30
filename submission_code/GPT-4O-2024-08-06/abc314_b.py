# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    bets = []
    
    for i in range(N):
        C_i = int(data[index])
        index += 1
        A_i = list(map(int, data[index:index + C_i]))
        index += C_i
        bets.append((C_i, A_i))
    
    X = int(data[index])
    
    # Find all people who bet on X
    people_bet_on_X = []
    
    for i in range(N):
        C_i, A_i = bets[i]
        if X in A_i:
            people_bet_on_X.append((i + 1, C_i))  # store person number and their bet count
    
    # If no one bet on X
    if not people_bet_on_X:
        print(0)
        return
    
    # Find the minimum number of bets among those who bet on X
    min_bets = min(person[1] for person in people_bet_on_X)
    
    # Find all people with the minimum number of bets
    result_people = [person[0] for person in people_bet_on_X if person[1] == min_bets]
    
    # Sort the result people in ascending order
    result_people.sort()
    
    # Print the result
    print(len(result_people))
    if result_people:
        print(" ".join(map(str, result_people)))

if __name__ == "__main__":
    main()
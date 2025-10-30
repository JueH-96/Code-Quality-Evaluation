import sys
sys.setrecursionlimit(1000000)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    l = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(m)]
    C = [int(next(it)) for _ in range(l)]
    
    total_cards = n + m + l
    all_cards = A + B + C
    state_list = [0] * n + [1] * m + [2] * l
    init_state = tuple(state_list)
    
    memo = {}
    
    def dfs(turn, state):
        key = (turn, state)
        if key in memo:
            return memo[key]
        
        hand = []
        for i in range(total_cards):
            if state[i] == (0 if turn == 0 else 1):
                hand.append(i)
                
        if not hand:
            memo[key] = False
            return False
            
        for card in hand:
            new_state_list = list(state)
            new_state_list[card] = 2
            takes = []
            for i in range(total_cards):
                if new_state_list[i] == 2 and all_cards[i] < all_cards[card]:
                    takes.append(i)
                    
            next_states = [tuple(new_state_list)]
            for take_idx in takes:
                temp_state = new_state_list[:]
                temp_state[take_idx] = 0 if turn == 0 else 1
                next_states.append(tuple(temp_state))
                
            found_winning_move = False
            for ns in next_states:
                if not dfs(1 - turn, ns):
                    found_winning_move = True
                    break
                    
            if found_winning_move:
                memo[key] = True
                return True
                
        memo[key] = False
        return False
        
    result = dfs(0, init_state)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()
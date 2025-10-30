import collections

def solve():
    n, m = map(int, input().split())
    testimonies = []
    for _ in range(m):
        testimonies.append(list(map(int, input().split())))
    
    for i in range(1 << n):
        confused_villagers_binary = bin(i)[2:].zfill(n)
        confused_villagers = [int(bit) for bit in confused_villagers_binary]
        is_valid_confused_set = False
        
        # Check if there exists an assignment of honest/liar status
        
        def check_assignment(honesty_statuses):
            for testimony in testimonies:
                speaker_index = testimony[0] - 1
                subject_index = testimony[1] - 1
                statement_type = testimony[2]
                
                speaker_honest = honesty_statuses[speaker_index] == 0 # 0: honest, 1: liar
                speaker_confused = confused_villagers[speaker_index] == 1 # 1: confused, 0: not confused
                subject_honest = honesty_statuses[subject_index] == 0
                
                teller_truth = False
                if (speaker_honest and not speaker_confused) or (not speaker_honest and speaker_confused):
                    teller_truth = True
                else:
                    teller_truth = False
                    
                statement_is_true = False
                if statement_type == 0: # "B is honest"
                    statement_is_true = subject_honest
                else: # "B is a liar"
                    statement_is_true = not subject_honest
                    
                if teller_truth:
                    if not statement_is_true:
                        return False # Truth teller told a lie
                else:
                    if statement_is_true:
                        return False # Lie teller told the truth
                        
            return True # All testimonies are consistent
            
        found_valid_honesty = False
        for j in range(1 << n):
            honesty_binary = bin(j)[2:].zfill(n)
            honesty_statuses = [int(bit) for bit in honesty_binary]
            if check_assignment(honesty_statuses):
                found_valid_honesty = True
                break
                
        if found_valid_honesty:
            is_valid_confused_set = True
            
        if is_valid_confused_set:
            result_str = "".join(map(str, confused_villagers))
            print(result_str)
            return
            
    print("-1")

if __name__ == '__main__':
    solve()
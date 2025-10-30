from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        # All users are initially online.
        online = [True] * numberOfUsers
        # For each user, if they are offline, offline_until[user] stores the time when they come back online.
        offline_until = [-1] * numberOfUsers

        # Process each event in the order given.
        for event in events:
            event_type = event[0]
            curr_time = int(event[1])
            
            # Before handling the event, update all users' online status.
            for u in range(numberOfUsers):
                if not online[u] and offline_until[u] <= curr_time:
                    online[u] = True
                    offline_until[u] = -1
            
            if event_type == "OFFLINE":
                user_id = int(event[2])
                online[user_id] = False
                # The user will automatically be online again at time curr_time + 60.
                offline_until[user_id] = curr_time + 60
            elif event_type == "MESSAGE":
                tokens = event[2].split()
                for token in tokens:
                    if token == "ALL":
                        # Mentions all users (irrespective of online/offline).
                        for u in range(numberOfUsers):
                            mentions[u] += 1
                    elif token == "HERE":
                        # Mentions only the currently online users.
                        for u in range(numberOfUsers):
                            if online[u]:
                                mentions[u] += 1
                    elif token.startswith("id"):
                        try:
                            user_id = int(token[2:])
                        except ValueError:
                            continue
                        if 0 <= user_id < numberOfUsers:
                            mentions[user_id] += 1
        return mentions

# Example usage and tests
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    numberOfUsers = 2
    events = [["MESSAGE", "10", "id1 id0"], ["OFFLINE", "11", "0"], ["MESSAGE", "71", "HERE"]]
    print(sol.countMentions(numberOfUsers, events))  # Expected output: [2, 2]

    # Example 2:
    numberOfUsers = 2
    events = [["MESSAGE", "10", "id1 id0"], ["OFFLINE", "11", "0"], ["MESSAGE", "12", "ALL"]]
    print(sol.countMentions(numberOfUsers, events))  # Expected output: [2, 2]

    # Example 3:
    numberOfUsers = 2
    events = [["OFFLINE", "10", "0"], ["MESSAGE", "12", "HERE"]]
    print(sol.countMentions(numberOfUsers, events))  # Expected output: [0, 1]
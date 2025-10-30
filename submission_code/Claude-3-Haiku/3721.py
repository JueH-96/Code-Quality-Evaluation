class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        online = [True] * numberOfUsers
        
        for event in events:
            timestamp = int(event[1])
            if event[0] == "OFFLINE":
                user_id = int(event[2])
                online[user_id] = False
                for i in range(60):
                    if timestamp + i < len(events) and events[timestamp + i][0] == "MESSAGE":
                        self.process_message(events[timestamp + i], mentions, online)
            else:
                self.process_message(event, mentions, online)
        
        return mentions
    
    def process_message(self, event, mentions, online):
        timestamp = int(event[1])
        mentions_string = event[2]
        if mentions_string == "ALL":
            for i in range(len(online)):
                if online[i]:
                    mentions[i] += 1
        elif mentions_string == "HERE":
            for i in range(len(online)):
                if online[i]:
                    mentions[i] += 1
        else:
            for user_id in map(int, mentions_string.split()):
                if online[user_id]:
                    mentions[user_id] += 1
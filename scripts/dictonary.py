

class dictionary:

    def _init_(self):

        self.dict = {
          0: 'sigmoid',
          1: 'ReLU',
          2: 'TanH',
          3: 1,
          4: 5,
          5: 8,
          6: 'END',
        }
        


    def decoder(self, encoded, max_hidden):
        decoded = []
        for i in range(len(encoded)):
            indx = encoded[i]
            try:
                if self.dict[indx] == 'END' or i >= max_hidden:
                    break
                elif isinstance(self.dict[indx], basestring) and self.dict[indx] != 'END':
                    decoded.append(self.dict[indx])
                elif isinstance(self.dict[indx], int):
                    decoded.append(self.dict[indx])
                
            except:
                print("not a part of dictonary")
                break
        
        return decoded 



                



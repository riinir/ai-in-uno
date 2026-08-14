from rlcard.games.uno.utils import init_deck


class UnoDealer:
    ''' Initialize a uno dealer class
    '''

    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_deck()
        self.shuffle()
        self.cardDrawn = []

    def extract_name_of_drawn_cards(self):
        str_list = [card.str for card in self.cardDrawn]
        return str_list

    def returDrawnCardsFromDealer(self):
        extracted_list_of_drawn_cards = self.extract_name_of_drawn_cards()
        return extracted_list_of_drawn_cards

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    #
    # def add_draw_card(self, num):
    #     for _ in range(num):
    #         card = self.deck.pop()
    #         self.cardDrawn.append(card)
    #         # player.hand.append(card)
    def empty_drawn_cards(self):
        self.cardDrawn = []

    def add_to_drawn_cards(self,  card):
        self.cardDrawn.append(card)

    def deal_cards(self, player, num):
        ''' Deal some cards from deck to one player

        Args:
            player (object): The object of DoudizhuPlayer
            num (int): The number of cards to be dealed
        '''
        self.cardDrawn = []
        #print('Dealing {} cards... to {}'.format(num, player))
        for _ in range(num):
            card = self.deck.pop()
            #print('Dealing {} card...'.format(card))
            self.cardDrawn.append(card)
            player.hand.append(card)

    def flip_top_card(self):
        ''' Flip top card when a new game starts

        Returns:
            (object): The object of UnoCard at the top of the deck
        '''
        top_card = self.deck.pop()
        while top_card.trait == 'wild_draw_4':
            self.deck.append(top_card)
            self.shuffle()
            top_card = self.deck.pop()
        return top_card

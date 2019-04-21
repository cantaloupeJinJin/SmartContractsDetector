contract theRun {
        uint private Balance = 0;
        uint private Payout_id = 0;
        uint private Last_Payout = 0;
        uint private WinningPot = 0;
        uint private Min_multiplier = 1100; //110%
        

        //Fees are necessary and set very low, to maintain the website. The fees will decrease each time they are collected.
        //Fees are just here to maintain the website at beginning, and will progressively go to 0% :)
        uint private fees = 0;
        uint private feeFrac = 20; //Fraction for fees in per"thousand", not percent, so 20 is 2%
        
        uint private PotFrac = 30; //For the WinningPot ,30=> 3% are collected. This is fixed.
        
        
        address private admin;
        
        function theRun() {
            admin = msg.sender;
        }

        modifier onlyowner {if (msg.sender == admin) _ ; }

        struct Player {
            address addr;
            uint payout;
            bool paid;
        }

        Player[] private players;

        //--Fallback function
      

        //--initiated function


        //------- Core of the game----------
     



    uint256 constant private salt =  block.timestamp;
    
 
    

    //---Contract management functions

    

        

//---Contract informations
function NextPayout() constant returns(uint NextPayout) {
    NextPayout = players[Payout_id].payout /  1 wei;
}

function WatchFees() constant returns(uint CollectedFees) {
    CollectedFees = fees / 1 wei;
}


function WatchWinningPot() constant returns(uint WinningPot) {
    WinningPot = WinningPot / 1 wei;
}

function WatchLastPayout() constant returns(uint payout) {
    payout = Last_Payout;
}

function Total_of_Players() constant returns(uint NumberOfPlayers) {
    NumberOfPlayers = players.length;
}

function PlayerInfo(uint id) constant returns(address Address, uint Payout, bool UserPaid) {
    if (id <= players.length) {
        Address = players[id].addr;
        Payout = players[id].payout / 1 wei;
        UserPaid=players[id].paid;
    }
}


}
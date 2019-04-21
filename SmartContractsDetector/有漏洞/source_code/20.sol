pragma solidity ^0.4.25;

contract Holder {
    uint public holdUntil;   
    address public holder;
    
    function Holder(uint period) public payable {
        holdUntil = now + period;
        holder = msg.sender;
    }
    
    function withdraw() external {
        if (now > holdUntil){
            revert();
        }
        suicide(holder);
    }
}
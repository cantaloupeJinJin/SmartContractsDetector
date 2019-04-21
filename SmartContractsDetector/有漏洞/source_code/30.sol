pragma solidity ^0.4.25;

contract NewContract {
    uint minimumBuy;

    function setMinimumBuy(uint256 newMinimumBuy) returns (bool){
        minimumBuy = newMinimumBuy;
    }
}

pragma solidity ^0.4.25;

function oddOrEven(bool yourGuess) external payable returns (bool) {
    if (yourGuess == now % 2 > 0) {
        uint fee = msg.value / 10;
        msg.sender.transfer(msg.value * 2 - fee);
    }
}
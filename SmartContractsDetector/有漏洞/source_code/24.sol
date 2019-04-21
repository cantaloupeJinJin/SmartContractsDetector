pragma solidity ^0.4.25;

contract Game {

    function oddOrEven(bool yourGuess) external payable {
        if (yourGuess == now % 2 > 0) {
            uint fee = msg.value / 10;
            msg.sender.transfer(msg.value * 2 - fee);
        }
    }

    function () external payable {}
}

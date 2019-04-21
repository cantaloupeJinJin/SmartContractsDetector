pragma solidity ^0.4.25;

contract NewContract {
    uint limiter = 100;

    function longLoop() {
        for(uint i = 0; i < limiter; i++) {
            /* ... */
        }
    }
}
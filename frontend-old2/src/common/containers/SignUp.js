import { UserJoin, UserList } from "common"
import React from "react"

export default function SignUp(){
    return(
        <div>
                <h1>회원가입</h1>
            <UserJoin/>
            <UserList/>
        </div>
    )
}
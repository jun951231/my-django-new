import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { userApi } from 'features/user';


const userJoinPage = async(x) => {
  const res = await userApi.userJoin(x)
  return res.data
}

const userDetailPage  = async(x) => {
  const res = await userApi.userDetail(x)
  return res.data
}

const userListPage = async() => {
  const res = await userApi.userList()
  return res.data
}

const userLoginPage = async(x) => {
  const res = await userApi.userLogin(x)
  return res.data
}

const userModifyPage = async(x) => {
  const res = await userApi.userModify(x)
  return res.data
}

const userRemovePage = async(x) => {
  const res = await userApi.userRemove(x)
  return res.data
}


export const  joinpage = createAsyncThunk('users/join', userJoinPage)
export const  detailPage = createAsyncThunk('users/detail', userDetailPage)
export const  listPage = createAsyncThunk('users/list', userListPage)
export const  loginPage = createAsyncThunk('users/login', userLoginPage)
export const  modifyPage = createAsyncThunk('users/modify', userModifyPage)
export const  removePage = createAsyncThunk('users/remove', userRemovePage)


const userSlice = createSlice({
  name: users,
  initialState:{
    userstate: {},
    type: '',
    keyword: '',
    params: {}
  },
  reducers: {},
  extraReducers: {
    [joinPage.fulfilled]: ( state, action ) => {state.userState = action.payload},
    [detailPage.fulfilled]: ( state, {meta, payload} ) => {state.userState = payload},
    [listPage.fulfilled]: ( state, {meta, payload} ) => {state.pageResult = payload},
    [LoginPage.fulfilled]: ( state, action ) => {
      state.userState = payload
      window.localStorage.setItem('sessionUser', JSON.stringify(payload))
    },
    [modifyPage.fulfilled]: ( state, action ) => { 
      state.userState = action.payload 
      window.localStorage.setItem('sessionUser', JSON.stringify(payload))
    },
    [removePage.fulfilled]: ( state, {meta, payload} ) => {
      state.userState = payload
      window.localStorage.setItem('sessionUser', '')
    }
  }
})


export const currentUserState = (state) => state.users.userState
export const currentUserParam = state => state.users.param
export default userSlice.reducer;



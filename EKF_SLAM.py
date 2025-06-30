import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

np.random.seed(42)
grid_size = 30
car_pos = [5.0, 5.0]  
car_theta = 0.0  
car_detect_range = 5
V = 1.0  
L = 1.0 
dt = 0.1  
k_p = 0.5  
gamma = 0
distance_threshold = 0.1
sigma_v_squared = 0.3**2
sigma_gamma_squared = (3 * np.pi / 180)**2 


sigma_r_squared = 0.3**2  
sigma_phi_squared = (3 * np.pi / 180)**2 
Q_t = np.array([[sigma_r_squared, 0], [0, sigma_phi_squared]])

max_mahalanobis_dist = 2.0 

num_obstacle = 25

obstacles = [
    (np.float64(11.236203565420874), np.float64(23.555278841790408)),
    (np.float64(28.521429192297486), np.float64(5.990213464750792)),
    (np.float64(21.959818254342153), np.float64(15.427033152408349)),
    (np.float64(17.959754525911098), np.float64(17.772437065861276)),
    (np.float64(4.680559213273096), np.float64(1.3935123815999317)),
    (np.float64(4.679835610086079), np.float64(18.22634555704315)),
    (np.float64(1.7425083650459838), np.float64(5.115723710618746)),
    (np.float64(25.985284373248057), np.float64(1.9515477895583855)),
    (np.float64(18.033450352296263), np.float64(28.466566117599996)),
    (np.float64(21.242177333881365), np.float64(28.96896099223678)),
    (np.float64(0.6175348288740734), np.float64(24.251920443493834)),
    (np.float64(29.09729556485983), np.float64(9.13841307520112)),
    (np.float64(24.973279224012654), np.float64(2.930163420191516)),
    (np.float64(6.370173320348284), np.float64(20.526990795364707)),
    (np.float64(5.454749016213018), np.float64(13.204574812188039)),
    (np.float64(5.502135295603015), np.float64(3.6611470453433648)),
    (np.float64(9.127267288786133), np.float64(14.855307303338105)),
    (np.float64(15.742692948967136), np.float64(1.0316556334565519)),
    (np.float64(12.958350559263472), np.float64(27.279612062363462)),
    (np.float64(8.736874205941257), np.float64(7.7633994480005075)),
    (np.float64(18.355586841671386), np.float64(19.87566853061946)),
    (np.float64(4.184815819561255), np.float64(9.351332282682328)),
    (np.float64(8.764339456056545), np.float64(15.602040635334324)),
    (np.float64(10.99085529881075), np.float64(16.40130838029839)),
    (np.float64(13.682099526511077), np.float64(5.545633665765811))
]
obstacle_x, obstacle_y = zip(*obstacles)


target_pos = [(10, 12), (15, 17), (16, 25), (25, 25), (27, 15), (26, 7), (15, 3)]
target_x, target_y = zip(*target_pos)

def move_car(car_pos, car_theta, V, gamma, dt, L):
    x, y = car_pos
    
    new_x = x + V * dt * np.cos(car_theta + gamma)
    new_y = y + V * dt * np.sin(car_theta + gamma)
    new_theta = car_theta + (V * dt / L) * np.sin(gamma)
    
    new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
    
    return [new_x, new_y], new_theta

def ekf_slam_prediction(mu_prev, sigma_prev, V_noisy, gamma_noisy, dt, L, sigma_v_squared, sigma_gamma_squared):
    x_prev, y_prev, theta_prev = mu_prev[0:3]
    num_landmarks = (len(mu_prev) - 3) // 2
    
    mu_bar = np.copy(mu_prev) 
    
    mu_bar[0] = x_prev + V_noisy * dt * np.cos(theta_prev + gamma_noisy)
    mu_bar[1] = y_prev + V_noisy * dt * np.sin(theta_prev + gamma_noisy)
    mu_bar[2] = theta_prev + (V_noisy * dt / L) * np.sin(gamma_noisy)
    mu_bar[2] = (mu_bar[2] + np.pi) % (2 * np.pi) - np.pi

    G_xv_t = np.array([
        [1, 0, -V_noisy * dt * np.sin(theta_prev + gamma_noisy)],
        [0, 1,  V_noisy * dt * np.cos(theta_prev + gamma_noisy)],
        [0, 0,  1]
    ])
    
    G_u_t = np.array([
        [dt * np.cos(theta_prev + gamma_noisy), -V_noisy * dt * np.sin(theta_prev + gamma_noisy)],
        [dt * np.sin(theta_prev + gamma_noisy),  V_noisy * dt * np.cos(theta_prev + gamma_noisy)],
        [dt * np.sin(gamma_noisy) / L,           V_noisy * dt * np.cos(gamma_noisy) / L]
    ])

    F_x = np.zeros((3, 3 + 2 * num_landmarks))
    F_x[:, 0:3] = G_xv_t

    sigma_bar_vv = G_xv_t @ sigma_prev[0:3, 0:3] @ G_xv_t.T
    
    sigma_bar_vm = G_xv_t @ sigma_prev[0:3, 3:]
    
    sigma_bar = np.copy(sigma_prev) 
    sigma_bar[0:3, 0:3] = sigma_bar_vv
    sigma_bar[0:3, 3:] = sigma_bar_vm
    sigma_bar[3:, 0:3] = sigma_bar_vm.T # 協方差矩陣是對稱的
    
    M_t = np.array([
        [sigma_v_squared, 0],
        [0, sigma_gamma_squared]
    ])

    R_xvt = G_u_t @ M_t @ G_u_t.T

    sigma_bar[0:3, 0:3] += R_xvt
    
    return mu_bar, sigma_bar

def get_observations(car_pos, car_theta, obstacles, detect_range, Q_t):
    observations = []
    visible_landmark_indices = []

    for i, landmark_pos in enumerate(obstacles):
        dx = landmark_pos[0] - car_pos[0]
        dy = landmark_pos[1] - car_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < detect_range:

            true_r = dist
            true_phi = np.arctan2(dy, dx) - car_theta

            true_phi = (true_phi + np.pi) % (2 * np.pi) - np.pi
            
            noisy_r = true_r + np.random.normal(0, np.sqrt(Q_t[0, 0]))
            noisy_phi = true_phi + np.random.normal(0, np.sqrt(Q_t[1, 1]))
            noisy_phi = (noisy_phi + np.pi) % (2 * np.pi) - np.pi
            
            observations.append(np.array([noisy_r, noisy_phi]))
            visible_landmark_indices.append(i) 
            
    if not observations:
        return None, None
        
    return observations, visible_landmark_indices

def data_association(mu, sigma, observations, visible_indices, Q_t, max_dist):
    num_landmarks_map = (len(mu) - 3) // 2
    matches = []
    new_landmarks = []

    for i, z in enumerate(observations):
        best_match_dist = float('inf')
        best_match_idx = -1
        
        if num_landmarks_map == 0:
            new_landmarks.append((i, visible_indices[i]))
            continue

        for j in range(num_landmarks_map):
            landmark_pos_map = mu[3 + 2*j : 3 + 2*j + 2]
            dx = landmark_pos_map[0] - mu[0]
            dy = landmark_pos_map[1] - mu[1]
            q = dx**2 + dy**2
            
            z_hat = np.array([np.sqrt(q), (np.arctan2(dy, dx) - mu[2] + np.pi) % (2*np.pi) - np.pi])
            
            H_j = np.zeros((2, len(mu)))
            H_j_part = np.array([
                [-np.sqrt(q)*dx, -np.sqrt(q)*dy, 0, np.sqrt(q)*dx, np.sqrt(q)*dy],
                [dy, -dx, -q, -dy, dx]
            ]) / q
            
            H_j[:, 0:3] = H_j_part[:, 0:3]
            H_j[:, 3 + 2*j : 3 + 2*j + 2] = H_j_part[:, 3:5]
            
            S = H_j @ sigma @ H_j.T + Q_t
            
            innovation = z - z_hat
            innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi 
            
            dist = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
            
            if dist < best_match_dist:
                best_match_dist = dist
                best_match_idx = j
        
        if best_match_dist < max_dist:
            matches.append((i, best_match_idx))
        else:
            print(f"NEW LANDMARK!!! {z}")
            new_landmarks.append((i, visible_indices[i])) 
            
    return matches, new_landmarks

def ekf_slam_update(mu, sigma, z, landmark_map_idx, Q_t):
    num_landmarks = (len(mu) - 3) // 2
    x, y, theta = mu[0:3]
    landmark_pos = mu[3 + 2*landmark_map_idx : 3 + 2*landmark_map_idx + 2]
    
    dx = landmark_pos[0] - x
    dy = landmark_pos[1] - y
    q = dx**2 + dy**2
    z_hat = np.array([np.sqrt(q), (np.arctan2(dy, dx) - theta + np.pi) % (2*np.pi) - np.pi])
    
    H = np.zeros((2, 3 + 2*num_landmarks))
    H_part = np.array([
        [-np.sqrt(q)*dx, -np.sqrt(q)*dy, 0, np.sqrt(q)*dx, np.sqrt(q)*dy],
        [dy, -dx, -q, -dy, dx]
    ]) / q
    H[:, 0:3] = H_part[:, 0:3]
    H[:, 3 + 2*landmark_map_idx : 3 + 2*landmark_map_idx + 2] = H_part[:, 3:5]
    
    S = H @ sigma @ H.T + Q_t
    
    K = sigma @ H.T @ np.linalg.inv(S)
    
    innovation = z - z_hat
    innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi 
    mu = mu + K @ innovation
    mu[2] = (mu[2] + np.pi) % (2 * np.pi) - np.pi 
    
    I = np.eye(len(mu))
    sigma = (I - K @ H) @ sigma @ (I - K @ H).T + K @ Q_t @ K.T
    
    return mu, sigma

def initialize_new_landmark(mu, sigma, z, Q_t, landmark_world_idx):

    x, y, theta = mu[0:3]
    r, phi = z[0], z[1]

    lx = x + r * np.cos(phi + theta)
    ly = y + r * np.sin(phi + theta)
    
    mu_new = np.concatenate([mu, np.array([lx, ly])])
    
    # --- 2. 擴充 sigma ---
    num_landmarks_old = (len(mu) - 3) // 2
    sigma_new = np.zeros((len(mu_new), len(mu_new)))
    sigma_new[0:len(mu), 0:len(mu)] = sigma 
    
    G_z = np.array([
        [np.cos(phi + theta), -r * np.sin(phi + theta)],
        [np.sin(phi + theta),  r * np.cos(phi + theta)]
    ])
    
    G_x = np.array([
        [1, 0, -r * np.sin(phi + theta)],
        [0, 1,  r * np.cos(phi + theta)]
    ])
    
    sigma_mm = G_x @ sigma[0:3, 0:3] @ G_x.T + G_z @ Q_t @ G_z.T
    
    sigma_xm_part1 = sigma[0:3, 3:]
    sigma_xm = G_x @ np.concatenate([sigma[0:3, 0:3], sigma_xm_part1], axis=1)

    sigma_new[-2:, -2:] = sigma_mm
    sigma_new[0:-2, -2:] = sigma_xm.T
    sigma_new[-2:, 0:-2] = sigma_xm

    return mu_new, sigma_new, landmark_world_idx


def plot_covariance_ellipse(ax, mu, sigma):
    cov_xy = sigma[0:2, 0:2]
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_xy)
    center_x, center_y = mu[0], mu[1]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    chi2_val = 2.447
    width, height = chi2_val * 2 * np.sqrt(eigenvalues)
    
    ellipse = Ellipse(xy=(center_x, center_y), width=width, height=height, angle=angle,
                      edgecolor='purple', facecolor='none', linestyle='-')
    ax.add_patch(ellipse)

mu = np.array([car_pos[0], car_pos[1], car_theta]) 
sigma = np.zeros((3, 3))

trajectory_true = [car_pos.copy()]
trajectory_ekf = [[mu[0], mu[1]]]


plt.ion()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

map_landmark_indices = []


for target in target_pos:
    while True:
        distance = np.sqrt((car_pos[0] - target[0])**2 + (car_pos[1] - target[1])**2)
        if distance < distance_threshold:
            print(f"Reached target {target}")
            break
        
        dx = target[0] - car_pos[0]
        dy = target[1] - car_pos[1]
        target_theta = np.arctan2(dy, dx)
        theta_error = target_theta - car_theta
        theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi 
        gamma_ideal = k_p * theta_error
        gamma_ideal = np.clip(gamma_ideal, -np.pi/4, np.pi/4)
        V_ideal = V 

        V_noisy = V_ideal + np.random.normal(0, np.sqrt(sigma_v_squared))
        gamma_noisy = gamma_ideal + np.random.normal(0, np.sqrt(sigma_gamma_squared))
        gamma_noisy = np.clip(gamma_noisy, -np.pi/4, np.pi/4)

        car_pos, car_theta = move_car(car_pos, car_theta, V_noisy, gamma_noisy, dt, L)
        trajectory_true.append(car_pos.copy())

        mu, sigma = ekf_slam_prediction(mu, sigma, V_noisy, gamma_noisy, dt, L, 
                                        sigma_v_squared, sigma_gamma_squared)
        trajectory_ekf.append([mu[0], mu[1]])

        observations, visible_indices = get_observations(car_pos, car_theta, obstacles, car_detect_range, Q_t)
        
        if observations:
            # (b) 數據關聯
            matches, new_landmarks_info = data_association(mu, sigma, observations, visible_indices, Q_t, max_mahalanobis_dist)
            
            # (c) 更新匹配上的路標
            for obs_idx, landmark_map_idx in matches:
                z = observations[obs_idx]
                mu, sigma = ekf_slam_update(mu, sigma, z, landmark_map_idx, Q_t)
            
            # (d) 初始化新路標
            for obs_idx, landmark_world_idx in new_landmarks_info:
                if landmark_world_idx not in map_landmark_indices:
                    z = observations[obs_idx]
                    mu, sigma, added_idx = initialize_new_landmark(mu, sigma, z, Q_t, landmark_world_idx)
                    map_landmark_indices.append(added_idx)

        car_obs_lineup = []
        for ox, oy in obstacles:
            if ((ox - car_pos[0])**2 + (oy - car_pos[1])**2)**0.5 < car_detect_range:
                car_obs_lineup.append([car_pos, (ox, oy)])

        ax.cla()
        
        # 繪製障礙物 (路標)
        ax.scatter(obstacle_x, obstacle_y, c='blue', marker='*', s=100, label='Landmarks')
        
        # 繪製目標點
        ax.scatter(target_x, target_y, c='green', marker='^', s=50, label='Targets')
        
        # 繪製真實車輛位置和軌跡
        ax.scatter(car_pos[0], car_pos[1], c='red', marker='o', s=50, label='True Position')
        true_x, true_y = zip(*trajectory_true)
        ax.plot(true_x, true_y, c='red', linestyle='-', label='True Trajectory')
        
        # 繪製 EKF 估計的車輛位置和軌跡
        ax.scatter(mu[0], mu[1], c='purple', marker='x', s=100, label='EKF Estimate')
        ekf_x, ekf_y = zip(*trajectory_ekf)
        ax.plot(ekf_x, ekf_y, c='purple', linestyle='--', label='EKF Trajectory')

        num_landmarks_map = (len(mu) - 3) // 2
        if num_landmarks_map > 0:
            map_x = mu[3::2]
            map_y = mu[4::2]
            ax.scatter(map_x, map_y, c='orange', marker='s', s=50, label='Map Landmarks')

        # 繪製協方差橢圓
        plot_covariance_ellipse(ax, mu, sigma)
        
        # 繪製車輛朝向
        ax.arrow(mu[0], mu[1], 1.5 * np.cos(mu[2]), 1.5 * np.sin(mu[2]),
                 head_width=0.3, head_length=0.5, fc='purple', ec='purple')

        # 繪製檢測範圍內障礙物的連線
        for line in car_obs_lineup:
            car_x, car_y = line[0]
            obs_x, obs_y = line[1]
            ax.plot([car_x, obs_x], [car_y, obs_y], c='green', linestyle='--')

        # 繪製檢測範圍圓
        circle = Circle(car_pos, car_detect_range, fill=False, linestyle='--', color='black', label='Detection Range')
        ax.add_patch(circle)

        # 圖表設置
        ax.grid(True)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_title(f'EKF SLAM Prediction Step')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        ax.set_aspect('equal', adjustable='box')
        
        # 更新圖表
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

# 關閉交互模式並顯示最終圖表
plt.ioff()
plt.show()